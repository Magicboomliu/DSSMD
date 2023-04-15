import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys
sys.path.append("../../..")
from utils.DCP.pytorch_version_quick.GuideFilter.guided_filter import GuidedFilter2d,FastGuidedFilter2d



# Get the dark channels
def get_dark_channels(img,kernel_size):
    b,c,h,w = img.shape    
    pad = nn.ReflectionPad2d(padding=kernel_size//2)
    img = pad(img)
    unfold_op = nn.Unfold(kernel_size=(kernel_size,kernel_size),padding=0)

    # get the minimal value of the given image
    local_patches = unfold_op(img)
    dc,dark_channels_index = torch.min(local_patches,dim=1,keepdim=True)
    dc = dc.view(b,1,h,w)
    
    return dc

# compute the atmosphere
def get_atmosphere(image_tensor,dark_channels,top_candidates_ratio,open_threshold=False):
    # search the top darkest channels's index, use the index to search in the image tensor 
    # to get the brightness value of the image_tensor
    b,_,h,w = dark_channels.shape
    top_candidates_nums = int(h*w*top_candidates_ratio) 
    dark_channels = dark_channels.view(b,1,-1)
    searchidx = torch.argsort(-dark_channels,dim=-1)[:,:,:top_candidates_nums]
    searchidx = searchidx.repeat(1,3,1)

    image_ravel = image_tensor.view(b,3,-1)

    value = torch.gather(image_ravel,dim=2,index=searchidx)
    
    airlight,image_index = torch.max(value,dim =-1,keepdim=True)

    airlight = airlight.squeeze(-1)
    
    # mammulay set to 220.
    if open_threshold:
        airlight = torch.clamp(airlight,max=220)


    return airlight
    
# Get Transmission
def get_transmission(image_tensor,airlight,omega,kernel_size,open_threshold=True):
 
    airlight = airlight.unsqueeze(-1).unsqueeze(-1)
    processed = image_tensor/airlight

    # print(omega * get_dark_channels(processed,kernel_size).mean())
    
    raw_t = 1.0 - omega * get_dark_channels(processed,kernel_size)

    
    if open_threshold:
        return torch.clamp(raw_t,min=0.2)
    
    return raw_t

# change the numpy image into a torch shape
# [H,W,3] to [1,3,H,W]
def image_numpy_to_tensor(image_np):
    image_data = torch.from_numpy(image_np) # [H,W,3]
    image_data = image_data.permute(2,0,1).unsqueeze(0)
    return image_data



# Guided Filter Pytorch
def soft_matting(image_tensor,raw_transmission,r=40,eps=1e-3):
    # Normalized the image.
    # image tensor : [B,3,H,W]
    b,c,h,w = image_tensor.shape
    guided_filter = GuidedFilter2d(radius=r,eps=eps)
    
    image_ravel = image_tensor.view(b,3,-1)
    image_min,_ = torch.min(image_ravel,dim=-1,keepdim=True)
    image_max,_ = torch.max(image_ravel,dim=-1,keepdim=True)
    image_min = image_min.unsqueeze(-1)
    image_max = image_max.unsqueeze(-1)
    
    normalized_image = (image_tensor - image_min) /(image_max-image_min)

    refined_transmission = guided_filter(raw_transmission,normalized_image)
    
    return refined_transmission


def recover_depth(transmission,beta=0.001):
    
    pass



def get_radiance(image_tensor,airlight,cur_transmission):
    
    # make sure the image data type is float.
    image_tensor = image_tensor.float()
    
    # current transmission shape is [B,1,H,W]
    tiledt = cur_transmission.repeat(1,3,1,1)
    airlight = airlight.unsqueeze(-1).unsqueeze(-1)
        
    dehazed_images = (image_tensor-airlight)*1.0/tiledt + airlight
    
    return dehazed_images





import skimage.io
import matplotlib.pyplot as plt

if __name__=="__main__":
    
    image_path = "../numpy_version/15.png"
    image_data = np.array(Image.open(image_path))
    image_data = np.asarray(image_data,dtype=np.float64)
    
    image_data_tensor = image_numpy_to_tensor(image_data)
    # print(image_data_tensor.shape)
    
    dark_channels  = get_dark_channels(img=image_data_tensor,kernel_size=15)
    
    
    # [B,3]---> atmosphere light 
    airlight = get_atmosphere(image_data_tensor,dark_channels,top_candidates_ratio=0.0001)
    
    
    # get the raw transmission
    raw_transmission = get_transmission(image_data_tensor,airlight=airlight,omega=0.95,kernel_size=15,open_threshold=True)

    print("raw_transmission: ",raw_transmission.mean())

    # raw transmission guided filtering
    refined_tranmission = soft_matting(image_data_tensor,raw_transmission,r=40,eps=1e-3)
    
    
    
    # np.save("refined_tranmission_torch.npy",refined_tranmission.squeeze(0).squeeze(0).cpu().numpy())
    
    
    # recover dehazing images.
    recovered_image = get_radiance(image_data_tensor,airlight,refined_tranmission)
    
    # # print(recovered_image.shape)
    plt.imshow(recovered_image.squeeze(0).permute(1,2,0).cpu().numpy()/255.)
    plt.show()
    
    
    
    
    # print(refined_tranmission.shape)
    
    # plt.imshow(refined_tranmission.squeeze(0).squeeze(0),cmap='gray')
    # plt.show()
    # print(raw_transmission.shape)
    
    
    # print(unfold.shape)
    # plt.imshow(image_data/255.)
    # plt.show()
    
    # plt.imshow(dark_channels.squeeze(0).squeeze(0),cmap='gray')
    # plt.show()
    # pass