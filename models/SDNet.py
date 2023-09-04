import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from models.UtilsNet.stereo_op import disp_warp,print_tensor_shape
from models.UtilsNet.cost_volume import CostVolume
from models.UtilsNet.disparity_estimation import DisparityEstimation
from models.UtilsNet.submoudles import *


# Deep Network for Simultaneous Stereo Matching and Dehazing.
class SDNet(nn.Module):
    def __init__(self,in_channels=3,max_disp=192,dehaze_switch=False):
        super(SDNet,self).__init__()
        self.in_channels = in_channels
        self.max_disp = max_disp
        self.dehaze_switch = dehaze_switch
        
        ''' Stereo Matching Branch Encoder Part '''
        # Stereo Macthing Branch Here
        self.conv1_sm = conv(3,32,7,2) # 1/2
        self.conv2_sm = ResBlock(32, 64, stride=2)            # 1/4
        self.conv3_sm = ResBlock(64, 128, stride=2)           # 1/8

        self.conv_redir_sm = ResBlock(128, 32, stride=1)    # skip connection
        # Mapping.
        self.conv3_1_sm = ResBlock(56, 128)
        # Further Downsample Stereo Branch
        self.conv4_sm = ResBlock(128, 256, stride=2)           # 1/16
        self.conv4_1_sm = ResBlock(256, 256)
        self.conv5_sm = ResBlock(256, 256, stride=2)           # 1/32
        self.conv5_1_sm = ResBlock(256, 256)
        self.conv6_sm = ResBlock(256, 512, stride=2)          # 1/64
        self.conv6_1_sm = ResBlock(512, 512)                  # 512 dimension


        ''' Dehazeing Branch Encoder Part '''
        self.conv1_dh = conv(3,32,7,2) # 1/2
        self.conv2_dh = ResBlock(32, 64, stride=2)            # 1/4
        self.conv3_dh = ResBlock(64, 128, stride=2)           # 1/8
        # Further Downsample Stereo Branch
        self.conv4_dh = ResBlock(128, 256, stride=2)           # 1/16
        self.conv4_1_dh = ResBlock(256, 256)
        self.conv5_dh = ResBlock(256, 256, stride=2)           # 1/32
        self.conv5_1_dh = ResBlock(256, 256)
        self.conv6_dh = ResBlock(256, 512, stride=2)          # 1/64
        self.conv6_1_dh = ResBlock(512, 512)                  # 512 dimension


        if self.dehaze_switch:
            self.upconv5_sm = deconv(512+512,256)
            self.upconv5_dh = deconv(512+512,256)
        else:
            self.upconv5_sm = deconv(512,256)

    

    
        ''' Stereo Matcing Branch Decoder '''    
        self.iconv5_sm = nn.ConvTranspose2d(512, 256, 3, 1, 1)
        self.iconv4_sm = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        self.iconv3_sm = nn.ConvTranspose2d(192, 64, 3, 1, 1)
        self.iconv2_sm = nn.ConvTranspose2d(96+1,32, 3, 1, 1)
        self.iconv1_sm = nn.ConvTranspose2d(64+1,32, 3, 1, 1)
        self.iconv0_sm = nn.ConvTranspose2d(19+1,16, 3, 1, 1)


        self.upconv4_sm = deconv(256, 128)
        self.upconv3_sm = deconv(128, 64)
        self.upconv2_sm = deconv(64, 32)
        self.upconv1_sm = deconv(32,32) # Note there is 32 dimension
        self.upconv0_sm = deconv(32,16)
        
        # disparity estimation branch
        self.disp3 = nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.disp2 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.disp1 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.disp0 = nn.Conv2d(16,1,kernel_size=3,stride=1,padding=1,bias=False)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu0 = nn.ReLU(inplace=True)


        ''' Transmission Estimation Decoder '''
        self.iconv5_dh = nn.ConvTranspose2d(512, 256, 3, 1, 1)
        self.iconv4_dh = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        self.iconv3_dh = nn.ConvTranspose2d(192, 64, 3, 1, 1)
        self.iconv2_dh = nn.ConvTranspose2d(96,32, 3, 1, 1)
        self.iconv1_dh = nn.ConvTranspose2d(64,32, 3, 1, 1)
        self.iconv0_dh = nn.ConvTranspose2d(19,16, 3, 1, 1)

        self.upconv4_dh = deconv(256, 128)
        self.upconv3_dh = deconv(128, 64)
        self.upconv2_dh = deconv(64, 32)
        self.upconv1_dh = deconv(32,32) # Note there is 32 dimension
        self.upconv0_dh = deconv(32,16)

        # transmission estimation branch
        self.transmision_estimation_branch = nn.Sequential(
                                                nn.Conv2d(16,1,3,1,1),
                                                nn.Sigmoid())


        '''Airlight Estimation Decoder'''
        # Airlight estimation
        self.conv1_air = conv(3,64,7,2) # 1/2
        self.conv2_air = conv(64,48,kernel_size=3,stride=2,batchNorm=True) # 1/4
        self.conv3_air = conv(48,32,kernel_size=3,stride=2,batchNorm=True) # 1/8
        self.conv4_air = conv(32,32,kernel_size=3,stride=2,batchNorm=True) # 1/16
        self.conv4_air = conv(32,16,kernel_size=3,stride=2,batchNorm=True) # 1/32

        self.airlight_head = nn.Sequential(nn.Linear(16,1,bias=True),
                                           nn.Sigmoid())


        
        

    def forward(self,haze_left,haze_right):
        
        haze_left = haze_left.float()
        haze_right = haze_right.float()

        '''Stereo Matching Feature Extractor '''
        # Feature Extractor Here: Left Image
        conv1_l_sm = self.conv1_sm(haze_left)          # 32 1/2
        conv2_l_sm = self.conv2_sm(conv1_l_sm)           # 64 1/4
        conv3_l_sm = self.conv3_sm(conv2_l_sm)           # 128 1/8
        # Feature Extractor Here: Right Images
        conv1_r_sm = self.conv1_sm(haze_right)
        conv2_r_sm = self.conv2_sm(conv1_r_sm)
        conv3_r_sm = self.conv3_sm(conv2_r_sm)           # 1/8
        # build cost volume here at the 1/8 resolution
        out_corr_sm = build_corr(conv3_l_sm,conv3_r_sm, self.max_disp//8) # 24 dimsion
        out_conv3a_redir_sm = self.conv_redir_sm(conv3_l_sm) # 32 dimension
        in_conv3b_sm = torch.cat((out_conv3a_redir_sm, out_corr_sm), dim=1)         # 24+32=56  1/8Resolution ---> 56 duemsnion
        # further downsample
        conv3b_sm = self.conv3_1_sm(in_conv3b_sm)    # 128
        conv4a_sm = self.conv4_sm(conv3b_sm)         
        conv4b_sm = self.conv4_1_sm(conv4a_sm)       # 256 1/16
        conv5a_sm = self.conv5_sm(conv4b_sm)
        conv5b_sm = self.conv5_1_sm(conv5a_sm)       # 256 1/32
        conv6a_sm = self.conv6_sm(conv5b_sm)
        conv6b_sm = self.conv6_1_sm(conv6a_sm)       # 512 1/64: Final Downsample Scale.

        '''Dehazing Network Feature Extractor'''
        conv1_l_dh = self.conv1_dh(haze_left)          # 32 1/2
        conv2_l_dh = self.conv2_dh(conv1_l_dh)           # 64 1/4
        conv3_l_dh = self.conv3_dh(conv2_l_dh)           # 128 1/8

        conv4a_dh = self.conv4_dh(conv3_l_dh)         
        conv4b_dh = self.conv4_1_dh(conv4a_dh)       # 256 1/16
        conv5a_dh = self.conv5_dh(conv4b_dh)
        conv5b_dh = self.conv5_1_dh(conv5a_dh)       # 256 1/32
        conv6a_dh = self.conv6_dh(conv5b_dh)
        conv6b_dh = self.conv6_1_dh(conv6a_dh)       # 512 1/64: Final Downsample Scale.

        # Fusion the dehaze feature and the stereo matcing feature
        if self.dehaze_switch:
            shared_decoder_feature = torch.cat((conv6b_sm,conv6b_dh),dim=1)
        else:
            shared_decoder_feature = conv6b_sm

        '''Stereo Estimation Branch Deocoder'''
        # Disparity Estimation at the 1/8 Resolution.
        upconv5_sm = self.upconv5_sm(shared_decoder_feature)      # 256 1/32        
        concat5_sm = torch.cat((upconv5_sm,conv5b_sm),dim=1)
        iconv5_sm = self.iconv5_sm(concat5_sm)       # 256
        upconv4_sm = self.upconv4_sm(iconv5_sm)      # 128 1/16
        concat4_sm = torch.cat((upconv4_sm, conv4b_sm), dim=1)   # 384 1/16
        iconv4_sm = self.iconv4_sm(concat4_sm)       # 256 1/16
        upconv3_sm = self.upconv3_sm(iconv4_sm)      # 64 1/8
        concat3_sm = torch.cat((upconv3_sm, conv3b_sm), dim=1)    # 64+128=192 1/8
        iconv3_sm = self.iconv3_sm(concat3_sm)       # 64
        # 1/8 Disparity Estimation.
        disp3 = self.disp3(iconv3_sm)
        disp3 = self.relu3(disp3)

        # Disparity Estimation at the 1/4 Resolution.
        disp2 = F.interpolate(disp3, size=(disp3.size()[2] * 2,disp3.size()[3] * 2), mode='bilinear') *2.0
        upconv2_sm = self.upconv2_sm(iconv3_sm)      # 32 1/4
        concat2_sm = torch.cat((upconv2_sm, conv2_l_sm,disp2), dim=1)  # 96 1/4
        iconv2_sm = self.iconv2_sm(concat2_sm) #32
        disp2 = self.disp2(iconv2_sm)
        disp2 = self.relu2(disp2)

         # Disparity Estimation at the 1/2 Resolution.
        disp1 = F.interpolate(disp2, size=(disp2.size()[2] * 2,disp2.size()[3] * 2), mode='bilinear') *2.0
        upconv1_sm = self.upconv1_sm(iconv2_sm) # 32 1/2
        concat1_sm = torch.cat((upconv1_sm,conv1_l_sm,disp1),dim=1) # 32+32 = 64+1
        iconv1_sm = self.iconv1_sm(concat1_sm) # 32
        disp1 = self.disp1(iconv1_sm)
        disp1 = self.relu1(disp1)

        # Disparity Estimation at the Full Resolution.
        disp0 = F.interpolate(disp1, size=(disp1.size()[2] * 2,disp1.size()[3] * 2), mode='bilinear') *2.0
        upconv0_sm = self.upconv0_sm(iconv1_sm) # 16 1/2
        concat0_sm = torch.cat((upconv0_sm,haze_left,disp0),dim=1) # 16+3 = 19+1
        iconv0_sm = self.iconv0_sm(concat0_sm) # 16
        disp0 = self.disp0(iconv0_sm)
        disp0 = self.relu0(disp0)

        '''Dehazing Estimation Branch Decoder'''
        if self.dehaze_switch:
            # transmission estimation.
            upconv5_dh = self.upconv5_dh(shared_decoder_feature)      # 256 1/32          
            concat5_dh = torch.cat((upconv5_dh,conv5b_dh),dim=1)
            iconv5_dh = self.iconv5_dh(concat5_dh)       # 256

            upconv4_dh = self.upconv4_dh(iconv5_dh)      # 128 1/16
            concat4_dh = torch.cat((upconv4_dh, conv4b_dh), dim=1)   # 384 1/16
            iconv4_dh = self.iconv4_dh(concat4_dh)       # 256 1/16
            
            upconv3_dh = self.upconv3_dh(iconv4_dh)      # 64 1/8
            concat3_dh = torch.cat((upconv3_dh, conv3_l_dh), dim=1)    # 64+128=192 1/8
            iconv3_dh = self.iconv3_dh(concat3_dh)       # 64

            upconv2_dh = self.upconv2_dh(iconv3_dh)      # 32 1/4
            concat2_dh = torch.cat((upconv2_dh, conv2_l_dh), dim=1)  # 96 1/4
            iconv2_dh = self.iconv2_dh(concat2_dh) #32
            
            upconv1_dh = self.upconv1_dh(iconv2_dh) # 32 1/2
            concat1_dh = torch.cat((upconv1_dh,conv1_l_dh),dim=1) # 32+32 = 64
            iconv1_dh = self.iconv1_dh(concat1_dh) # 32
            
            # full disparity estimation
            upconv0_dh = self.upconv0_dh(iconv1_dh) # 16 1/2
            concat0_dh = torch.cat((upconv0_dh,haze_left),dim=1) # 16+3 = 19
            iconv0_dh = self.iconv0_dh(concat0_dh) # 16
            
            predicted_transmission = self.transmision_estimation_branch(iconv0_dh)


            # airlight estimation.
            conv1_air = self.conv1_air(haze_left)
            conv2_air = self.conv2_air(conv1_air)
            conv3_air = self.conv3_air(conv2_air)
            conv4_air = self.conv4_air(conv3_air)
            airlight_feat = F.avg_pool2d(conv4_air, kernel_size=conv4_air.size()[2:]).squeeze(-1).squeeze(-1) #[B,1]
            estimated_airlight = self.airlight_head(airlight_feat) # (N,1)

        if self.dehaze_switch:
            return [disp3,disp2,disp1,disp0],predicted_transmission,estimated_airlight
        else:
            return [disp3,disp2,disp1,disp0]
        

            
        


if __name__=="__main__":
    
    left_input = torch.abs(torch.randn(8,3,320,640)).cuda()
    right_input = torch.abs(torch.randn(8,3,320,640)).cuda()
    
    model = SDNet(in_channels=3,dehaze_switch=True).cuda()

    disparity_pyramid,predicted_transmission,estimated_airlight = model(left_input,right_input)

    
    print_tensor_shape(disparity_pyramid)


