import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np



# Disparity Smooth-L1 Loss.
class Disparity_Loss(nn.Module):
    def __init__(self,type='smooth_l1',weights=[0.6,0.8,1.0,1.0]):
        super().__init__()
        self.loss_type = type
        if self.loss_type=='smooth_l1':
            self.disp_loss = nn.SmoothL1Loss(size_average=True,reduction='mean')
        self.weights = weights
    
    def forward(self,predicted_disparity_pyramid,gt_disparity):
        
        total_loss = 0
        # for multiple output
        if isinstance(predicted_disparity_pyramid,list) or isinstance(predicted_disparity_pyramid,tuple):
            for idx, predicted_disp in enumerate(predicted_disparity_pyramid):
                scale = gt_disparity.shape[-1]//predicted_disp.shape[-1]
                cur_gt_disparity = F.interpolate(cur_gt_disparity,scale_factor=1./scale,mode='bilinear',align_corners=False)*1.0/scale
                assert predicted_disp.shape[-1]==cur_gt_disparity.shape[-1]
                cur_loss = self.disp_loss(predicted_disp,cur_gt_disparity)
                total_loss+=self.weights[idx] * cur_loss
        # for single output
        else:
            assert predicted_disparity_pyramid.shape[-1] == gt_disparity.shape[-1]
            total_loss = self.disp_loss(predicted_disparity_pyramid,gt_disparity)

        return total_loss


# Transmission Map Loss.
class TransmissionMap_Loss(nn.Module):
    def __init__(self,type='smooth_l1'):
        super().__init__()
        self.trans_loss_type =type
        if self.trans_loss_type=='smooth_l1':
            self.trans_loss = nn.SmoothL1Loss(size_average=True,reduction='mean')
            
    def foward(self,predict_tranmission_map,gt_transmission_map):
        return self.trans_loss(predict_tranmission_map,gt_transmission_map)
        


# Airlight Loss.
class Airlight_Loss(nn.Module):
    def __init__(self,type='l1_loss'):
        super().__init__()
        self.arilight_loss_type = type
        if self.arilight_loss_type=='l1_loss':
            self.loss = nn.L1Loss(size_average=True,reduction='mean')
    def forward(self,pred_air,gt_air):
        return self.loss(pred_air,gt_air)



# recovered Clean Image Loss.
class RecoveredCleanImagesLoss(nn.Module):
    def __init__(self,type='normal'):
        super().__init__()
        self.loss_type = type
        self.loss_op = nn.L1Loss(size_average=True,reduction='mean')
        
    def forward(self,transmission_map,airlight,haze_image,clean_image):
        
        
        '''
        transmision: [B,1,H,W]
        haze image: [B,3,H,W]
        airlight: [B,1]
        '''
        airlight = airlight.unsqueeze(-1).unsqueze(-1) #[B,1,1,1]
        
        # if transmission = 0, how to real with?
        if transmission_map.min==0:
            recovered_clean = (haze_image-airlight*(1-transmission_map))/(transmission_map+1e-4)
        else:
            recovered_clean = (haze_image-airlight*(1-transmission_map))/(transmission_map)
            
        recovered_clean = torch.clamp(recovered_clean,min=0,max=1.0)
        
        if self.loss_type=='normal':
            loss = self.loss_op(recovered_clean,clean_image)

        return loss



# Recovered Depth Loss(Optional)
class RecoveredDepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        pass