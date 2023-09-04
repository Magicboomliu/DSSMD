import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np


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
                # upsample
                scale = gt_disparity.shape[-1]//predicted_disp.shape[-1]
                cur_gt_disparity = F.interpolate(gt_disparity,scale_factor=1./scale,mode='bilinear',align_corners=False)*1.0/scale
                
                assert predicted_disp.shape[-1]==cur_gt_disparity.shape[-1]
                cur_loss = self.disp_loss(predicted_disp,cur_gt_disparity)
                total_loss+=self.weights[idx] * cur_loss
        # for single output
        else:
            assert predicted_disparity_pyramid.shape[-1] == gt_disparity.shape[-1]
            total_loss = self.disp_loss(predicted_disparity_pyramid,gt_disparity)

        return total_loss
    

class TransmissionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,estimated_transmission,gt_transmission):
        return F.l1_loss(estimated_transmission,gt_transmission,size_average=True,reduction='mean')



class AirlightLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,estimated_airlight,gt_airlight):
        return F.l1_loss(estimated_airlight,gt_airlight,size_average=True,reduction='mean')


class Recover_Clean_Image_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,estimated_transmision,estimated_airlight,clean_images,haze_images):
        # estimated airlight # [N,1]
        H,W = estimated_transmision.shape[2:]
        batch_size = estimated_airlight.shape[0]
        estimated_airlight = estimated_airlight.view(batch_size,1,1,1).repeat(1,1,H,W)
        recovered_haze_images = estimated_transmision * clean_images + (1-estimated_transmision) * estimated_airlight

        recovered_haze_images = torch.clamp(recovered_haze_images,min=0,max=1.0)

        return F.l1_loss(recovered_haze_images,haze_images,size_average=True,reduction='mean')


class BMVC_Total_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.transmission_loss = TransmissionLoss()
        self.airlight_loss = AirlightLoss()
        self.rgb_loss = Recover_Clean_Image_Loss()
        
        self.disparity_loss = Disparity_Loss()
    
    def forward(self,pred_disp,gt_disp,
                estimated_transmision,estimated_airlight,clean_images,haze_images,gt_transmission,gt_airlight):
        # disparity Loss
        disp_loss = self.disparity_loss(pred_disp,gt_disp)
        # transmission loss
        transmission_loss = self.transmission_loss(estimated_transmision,gt_transmission)
        # airglight loss
        airlight_loss = self.airlight_loss(estimated_airlight,gt_airlight)
        # recovered RGB_loss
        rgb_loss = self.rgb_loss(estimated_transmision,estimated_airlight,clean_images,haze_images)

        total_loss = disp_loss + transmission_loss + airlight_loss + rgb_loss
        
        return total_loss,disp_loss,transmission_loss,airlight_loss,rgb_loss
    