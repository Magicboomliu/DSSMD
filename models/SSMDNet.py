import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from models.UtilsNet.stereo_op import disp_warp,print_tensor_shape
from models.UtilsNet.cost_volume import CostVolume
from models.UtilsNet.disparity_estimation import DisparityEstimation
from models.UtilsNet.submoudles import *



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)

# Pixel Attention: Spatial Attention
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

# Channel Attention
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid())
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

# One Blocks is Conv, PA Layer + CA Layer
class FeatureFusion(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(FeatureFusion, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x 
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x 
        return res





# Deep Network for Simultaneous Stereo Matching and Dehazing.
class SSMDNet(nn.Module):
    def __init__(self,in_channels=3,max_disp=192,dehaze_switch=False):
        super(SSMDNet,self).__init__()
        self.in_channels = in_channels
        self.max_disp = max_disp
        self.dehaze_switch = dehaze_switch

        '''Feature Fusion'''
        self.feature_fusion_module = FeatureFusion(conv=default_conv,dim=128,kernel_size=3)

        ''' Stereo Matching Branch Encoder Part '''
        # Stereo Macthing Branch Here
        self.conv1_sm = conv(3,32,7,2) # 1/2
        self.conv2_sm = ResBlock(32, 64, stride=2)            # 1/4
        self.conv3_sm = ResBlock(64, 128, stride=2)           # 1/8
        self.conv_redir_sm = ResBlock(128, 32, stride=1)    # skip connection
        # Mapping.
        self.conv3_1_sm = ResBlock(56, 128)

        '''Dehazing Branch Encoder Part'''
        self.conv1_dh = conv(3,32,7,2) # 1/2
        self.conv2_dh = ResBlock(32, 64, stride=2)            # 1/4
        self.conv3_dh = ResBlock(64, 128, stride=2)           # 1/8

        '''Shared U-Net For dehazing and stereo matcing'''
        self.conv4_shared = ResBlock(128, 256, stride=2)           # 1/16
        self.conv4_1_shared = ResBlock(256, 256)
        self.conv5_shared = ResBlock(256, 256, stride=2)           # 1/32
        self.conv5_1_shared = ResBlock(256, 256)
        self.conv6_shared = ResBlock(256, 512, stride=2)          # 1/64
        self.conv6_1_shared = ResBlock(512, 512)                  # 512 dimension

        self.iconv5_shared = nn.ConvTranspose2d(512, 256, 3, 1, 1)
        self.iconv4_shared = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        
        self.upconv5_shared = deconv(512,256)
        self.upconv4_shared = deconv(256, 128)


        '''Stereo Matching Decoder'''
        self.iconv3_sm = nn.ConvTranspose2d(192, 64, 3, 1, 1)
        self.iconv2_sm = nn.ConvTranspose2d(96+1,32, 3, 1, 1)
        self.iconv1_sm = nn.ConvTranspose2d(64+1,32, 3, 1, 1)
        self.iconv0_sm = nn.ConvTranspose2d(19+1,16, 3, 1, 1)
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



        '''transmission decoder'''
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



        '''Airlight estimation branch'''
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


        '''Dehazeing Feature Extractor'''
        conv1_l_dh = self.conv1_dh(haze_left)          # 32 1/2
        conv2_l_dh = self.conv2_dh(conv1_l_dh)           # 64 1/4
        conv3_l_dh = self.conv3_dh(conv2_l_dh)           # 128 1/8

        '''Feature Adaptive Fusion'''
        conv3b_sm_fusion = self.feature_fusion_module(conv3b_sm)
        conv3_l_dh_fusion = self.feature_fusion_module(conv3_l_dh)
        conv3b_stereo_dehaze_fusion = conv3b_sm_fusion + conv3_l_dh_fusion # [3,128,40,80]

        '''Shared U-Net Feature Aggregation'''
        conv4a_shared = self.conv4_shared(conv3b_stereo_dehaze_fusion)         
        conv4b_sharead = self.conv4_1_shared(conv4a_shared)       # 256 1/16
        conv5a_shared = self.conv5_shared(conv4b_sharead)
        conv5b_shared = self.conv5_1_shared(conv5a_shared)       # 256 1/32
        conv6a_shared = self.conv6_shared(conv5b_shared)
        conv6b_shared = self.conv6_1_shared(conv6a_shared)       # 512 1/64: Final Downsample Scale.

        upconv5_shared = self.upconv5_shared(conv6b_shared)      # 256 1/32        
        concat5_shared = torch.cat((upconv5_shared,conv5b_shared),dim=1)
        iconv5_share= self.iconv5_shared(concat5_shared)       # 256
        upconv4_shared = self.upconv4_shared(iconv5_share)      # 128 1/16
        concat4_shared = torch.cat((upconv4_shared, conv4b_sharead), dim=1)   # 384 1/16
        iconv4_shared = self.iconv4_shared(concat4_shared)       # 128 1/16

        '''Multi-Scale Stereo Matcing Estimation'''
        upconv3_sm = self.upconv3_sm(iconv4_shared)      # 64 1/8
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

        '''Transmission Estimation'''
        upconv3_dh = self.upconv3_dh(iconv4_shared)      # 64 1/8
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


        '''Airlight Estimation Head'''
        conv1_air = self.conv1_air(haze_left)
        conv2_air = self.conv2_air(conv1_air)
        conv3_air = self.conv3_air(conv2_air)
        conv4_air = self.conv4_air(conv3_air)
        airlight_feat = F.avg_pool2d(conv4_air, kernel_size=conv4_air.size()[2:]).squeeze(-1).squeeze(-1) #[B,1]
        estimated_airlight = self.airlight_head(airlight_feat) # (N,1)


        return [disp3,disp2,disp1,disp0],predicted_transmission,estimated_airlight
        

            
    
if __name__=="__main__":

    from losses.BMVC2018_Loss import BMVC_Total_Loss
    
    left_input = torch.abs(torch.randn(8,3,320,640)).cuda()
    right_input = torch.abs(torch.randn(8,3,320,640)).cuda()
    model = SSMDNet(in_channels=3,dehaze_switch=True).cuda()
    disparity_pyramid,predicted_transmission,estimated_airlight = model(left_input,right_input)

    haze_images = torch.abs(torch.randn(8,3,320,640)).cuda()

    gt_disp = torch.abs(torch.randn(8,1,320,640)).cuda()
    gt_tranmission = torch.sigmoid(torch.randn(8,1,320,640)).cuda()
    gt_airglight = torch.sigmoid(torch.randn(8,1)).cuda()

    
    bmvc_criten = BMVC_Total_Loss()

    total_loss,disp_loss,transmission_loss,airlight_loss,rgb_loss = bmvc_criten(pred_disp = disparity_pyramid,gt_disp=gt_disp,
                estimated_transmision= predicted_transmission,
                estimated_airlight= estimated_airlight,
                clean_images = left_input,
                haze_images= haze_images,
                gt_transmission= gt_tranmission,
                gt_airlight= gt_airglight)

    print(total_loss)
    # print_tensor_shape(disparity_pyramid)


