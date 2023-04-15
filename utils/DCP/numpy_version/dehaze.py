import numpy as np
from PIL import Image
import sys
sys.path.append("../../..")
from utils.DCP.numpy_version.guidedfiler import guided_filter
# from src.guidedfiler import guided_filter
import matplotlib.pyplot as plt


R,G,B = 0,1,2

L = 256 # color depth.


# Get the Dark Channel in the RGB images in local patch with a window size of `window_size`.
def get_dark_channel(input_rgb,window_size):
    '''
    Parameters: 
    ----------------
    Input_RGB: [H,W,3]
    windows_size: int.
    
    
    Return
    ----------
    [H,W] size dark channel prior for the input image `input_rgb`
    '''
    M, N, _ = input_rgb.shape
    padded = np.pad(input_rgb, ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + window_size, j:j + window_size, :])  # CVPR09, eq.5
    
    return darkch


def get_atmosphere(I, darkch, p):
    """Get the atmosphere light in the (RGB) image data.
    Parameters
    -----------
    I:      the M * N * 3 RGB image data ([0, L-1]) as numpy array
    darkch: the dark channel prior of the image as an M * N numpy array
    p:      percentage of pixels for estimating the atmosphere light
    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    # reference CVPR09, 4.4
    M, N = darkch.shape
    flatI = I.reshape(M * N, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[:int(M * N * p)]  # find top M * N * p indexes
    print('atmosphere light region:', [(i / N, i % N) for i in searchidx])

    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)


def get_transmission(I, A, darkch, omega, w):
    """Get the transmission esitmate in the (RGB) image data.
    Parameters
    -----------
    I:       the M * N * 3 RGB image data ([0, L-1]) as numpy array
    A:       a 3-element array containing atmosphere light
             ([0, L-1]) for each channel
    darkch:  the dark channel prior of the image as an M * N numpy array
    omega:   bias for the estimate = 0.95
    w:       window size for the estimate
    Return
    -----------
    An M * N array containing the transmission rate ([0.0, 1.0])
    """
    process = I /A
    # print("after processed mean: ", (1 - omega * get_dark_channel(process, w)).mean())

    return 1 - omega * get_dark_channel(process, w)  # CVPR09, eq.12



def get_radiance(I, A, cur_transmission):
    """Recover the radiance from raw image data with atmosphere light
       and transmission rate estimate.
    Parameters
    ----------
    I:      M * N * 3 data as numpy array for the hazy image
    A:      a 3-element array containing atmosphere light
            ([0, L-1]) for each channel
    t:      estimate fothe transmission rate
    Return
    ----------
    M * N * 3 numpy array for the recovered radiance
    """

    tiledt = np.stack((cur_transmission,cur_transmission,cur_transmission),axis=2)

    I = I.astype(np.float64)

    de_haze_image = (I - A) / tiledt +A # CVPR09, eq.16
    return de_haze_image
    # return (I - A) / tiledt +A # CVPR09, eq.16

import cv2

# def gamma_trans(img,gamma):#gamma函数处理
#     img = img/255
#     img = (img*255).astype(np.uint8)
#     gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]#建立映射表
#     gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)#颜色值为整数
#     return img
#     # return cv2.LUT(img,gamma_table)#图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

def recover_depth(transmission,beta=0.001):
    
    negative_depth = np.log(transmission)
    
    depth = negative_depth *-1 /beta
    
    return depth




import skimage.io

if __name__=="__main__":
    
    

    image_path = '15.png'
    image_data = np.array(Image.open(image_path))
    image_data = np.asarray(image_data,dtype=np.float64)
    
    # mammulay
    Amax= 220
    
    # get the dark channles
    dark_channels = get_dark_channel(image_data,window_size=15)
    
    # get the airlight of each channels.
    A = get_atmosphere(image_data,dark_channels,p=0.0001)
    
    A = np.minimum(A,Amax)
    
    # print(A.mean())
    
    raw_transmission = get_transmission(image_data,A,dark_channels,omega=0.95,w=15) 
    # suppose max t is equal to 0.2
    raw_transmission = np.maximum(raw_transmission,0.2)
    
    
    # guided filter: soft matting
    normI = (image_data - image_data.min()) / (image_data.max() - image_data.min())  # normalize I
    refined_transmission = guided_filter(normI, raw_transmission, r=40, eps=1e-3)


    
    depth = recover_depth(refined_transmission)
    
    recover_image = get_radiance(image_data,A,refined_transmission)
    
    
    skimage.io.imsave("recover.png",recover_image)
    # print(recover_image.max())
    # print(recover_image.min())
    # recover_image = gamma_trans(recover_image,gamma=0.8)
    # # recover_image = change_exposure(recover_image)
    # # # print(refined_transmission.dtype)
    # # recover_image = increase_brightness(recover_image)
    
    plt.subplot(2,3,1)
    plt.axis('off')
    plt.title("orginal images")
    plt.imshow(image_data.astype(np.uint8))
    plt.subplot(2,3,2)
    plt.axis('off')
    plt.title('dark channels')
    plt.imshow(dark_channels,cmap='gray')
    plt.subplot(2,3,3)
    plt.axis('off')
    plt.title('transmission raw')
    plt.imshow(raw_transmission,cmap='gray')
    plt.subplot(2,3,4)
    plt.title('transmission refined')
    plt.axis('off')
    plt.imshow(refined_transmission,cmap='gray')
    plt.subplot(2,3,5)
    plt.title('de-haze-image')
    plt.axis('off')
    plt.imshow(recover_image/255.0)
    plt.subplot(2,3,6)
    plt.title('scaled depth')
    plt.axis('off')
    plt.imshow(depth)
    plt.show()
    
    # return dark_channels,A, raw_transmission,refined_transmission
