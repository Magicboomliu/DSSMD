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
class DSSMMD(nn.Module):
    def __init__(self,in_channels=3,max_disp=192,dehaze_switch=False):
        super(DSSMMD,self).__init__()
        self.in_channels = in_channels
        self.max_disp = max_disp
        self.dehaze_switch = dehaze_switch
        
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
        self.conv6_1_sm = ResBlock(512, 512)
        
        # upsample
        self.iconv5_sm = nn.ConvTranspose2d(512, 256, 3, 1, 1)
        self.iconv4_sm = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        self.iconv3_sm = nn.ConvTranspose2d(192, 64, 3, 1, 1)
        self.iconv2_sm = nn.ConvTranspose2d(96,32, 3, 1, 1)
        self.iconv1_sm = nn.ConvTranspose2d(64,32, 3, 1, 1)
        self.iconv0_sm = nn.ConvTranspose2d(19,16, 3, 1, 1)
        
        
        self.upconv5_sm = deconv(512, 256)
        self.upconv4_sm = deconv(256, 128)
        self.upconv3_sm = deconv(128, 64)
        self.upconv2_sm = deconv(64, 32)
        self.upconv1_sm = deconv(32,32) # Note there is 32 dimension
        self.upconv0_sm = deconv(32,16)
        
        
        # disparity estimation 
        self.disp3 = nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu0 = nn.ReLU(inplace=True)
        
        # disparity residual
        self.residual_submodule_2 = res_submodule(scale=2,value_planes=32,out_planes=32)
        self.residual_submodule_1 = res_submodule(scale=1,value_planes=32,out_planes=32)
        self.residual_submodule_0 = res_submodule(scale=0,value_planes=16,out_planes=16)
        
    
    def forward(self,haze_left,haze_right):
        
        # Stereo Matching Branch
        conv1_l_sm = self.conv1_sm(haze_left)          # 32 1/2
        conv2_l_sm = self.conv2_sm(conv1_l_sm)           # 64 1/4
        conv3_l_sm = self.conv3_sm(conv2_l_sm)           # 128 1/8

        conv1_r_sm = self.conv1_sm(haze_right)
        conv2_r_sm = self.conv2_sm(conv1_r_sm)
        conv3_r_sm = self.conv3_sm(conv2_r_sm)           # 1/8

        # Build Correlation Cost Volume Here
        out_corr_sm = build_corr(conv3_l_sm,conv3_r_sm, self.max_disp//8)
        out_conv3a_redir_sm = self.conv_redir_sm(conv3_l_sm)
        in_conv3b_sm = torch.cat((out_conv3a_redir_sm, out_corr_sm), dim=1)         # 24+32=56  1/8Resolution
        
        # further downsample
        conv3b_sm = self.conv3_1_sm(in_conv3b_sm)    # 128
        conv4a_sm = self.conv4_sm(conv3b_sm)         
        conv4b_sm = self.conv4_1_sm(conv4a_sm)       # 256 1/16
        conv5a_sm = self.conv5_sm(conv4b_sm)
        conv5b_sm = self.conv5_1_sm(conv5a_sm)       # 256 1/32
        
        conv6a_sm = self.conv6_sm(conv5b_sm)
        conv6b_sm = self.conv6_1_sm(conv6a_sm)       # 512 1/64
        
        
        # Upsample
        upconv5_sm = self.upconv5_sm(conv6b_sm)      # 256 1/32        
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
        
        
        
        # 1/4 disparity estimation
        upconv2_sm = self.upconv2_sm(iconv3_sm)      # 32 1/4
        concat2_sm = torch.cat((upconv2_sm, conv2_l_sm), dim=1)  # 96 1/4
        iconv2_sm = self.iconv2_sm(concat2_sm) #32
        disp2 = F.interpolate(disp3, size=(disp3.size()[2] * 2,disp3.size()[3] * 2), mode='bilinear') *2.0
        res2 = self.residual_submodule_2(haze_left, haze_right, disp2, iconv2_sm)
        disp2 = disp2 + res2
        disp2 = self.relu2(disp2)
        
            
        # 1/2 disparity estimation
        upconv1_sm = self.upconv1_sm(iconv2_sm) # 32 1/2
        concat1_sm = torch.cat((upconv1_sm,conv1_l_sm),dim=1) # 32+32 = 64
        iconv1_sm = self.iconv1_sm(concat1_sm) # 32
        disp1 = F.interpolate(disp2, size=(disp2.size()[2] * 2,disp2.size()[3] * 2), mode='bilinear') *2.0
        res1 = self.residual_submodule_1(haze_left, haze_right, disp1, iconv1_sm)
        disp1 = disp1 + res1
        disp1 = self.relu1(disp1)
        
        
        # full disparity estimation
        upconv0_sm = self.upconv0_sm(iconv1_sm) # 16 1/2
        concat0_sm = torch.cat((upconv0_sm,haze_left),dim=1) # 16+3 = 19
        iconv0_sm = self.iconv0_sm(concat0_sm) # 16
        disp0 = F.interpolate(disp1, size=(disp1.size()[2] * 2,disp1.size()[3] * 2), mode='bilinear') *2.0
        res0 = self.residual_submodule_0(haze_left, haze_right, disp0, iconv0_sm)
        disp0 = disp0 + res0
        disp0 = self.relu0(disp0)
        
        
        
        
        
         
        
        


        
     




if __name__=="__main__":
    
    left_input = torch.abs(torch.randn(1,3,320,640)).cuda()
    right_input = torch.abs(torch.randn(1,3,320,640)).cuda()
    
    
    model = DSSMMD(in_channels=3).cuda()
    
    
    # Infernece
    model(left_input,right_input)
    
    pass