import cv2 as cv
import numpy as np
import math
from skimage.exposure import rescale_intensity
from varname import nameof
from roipoly import RoiPoly
from matplotlib import pyplot as plt

#can be varied/tuned according to desired extent of blending (sigmaFactor + const*sigmaStepSize)
sigmaStepSize = 0.65045

def imshow(img):

    windowName = nameof(img)
    cv.imshow(windowName,img.astype('float32'))
    k = cv.waitKey(0)
    cv.destroyAllWindows

    return

def ComputePyr(img, noLayers):

    downsample = img.copy()
    gPyr = [downsample]
    sigma = 0.7
    for i in range(noLayers):
        downsample = REDUCE(downsample,sigma)
        gPyr.append(downsample.astype('uint8'))
        sigma = sigma + sigmaStepSize
    lPyr = EXPAND(gPyr)

    return gPyr, lPyr

def REDUCE(img,sigma):

    blur = rgb_conv(img, GaussianK(sigma), 'zero')   #blurring
    return cv.merge((sample(blur[:,:,0], 1/2), sample(blur[:,:,1], 1/2), sample(blur[:,:,2], 1/2))) #downsample

def EXPAND(gPyr):

    numLevels = len(gPyr) -1
    lPyr = []
    lPyr.append(gPyr[numLevels - 1])
    #sigma must be tuned according to the resolution needed
    sigma =  0.475 + (numLevels + 1)*sigmaStepSize

    for i in range(numLevels - 1, 0, -1):

        #temp=gPyr[i]                       #upsample
        gExpand = cv.merge((sample(gPyr[i][:,:,0], 2),sample( gPyr[i][:,:,1], 2),sample( gPyr[i][:,:,2], 2)))
        gExpand = rgb_conv(gExpand, GaussianK(sigma), 'zero')
        laplacian = gPyr[i-1] - gExpand             
        lPyr.append(laplacian)
        sigma = sigma - sigmaStepSize

    return lPyr

def conv2(image,kernel,type):

    i_m=image.shape[0] #rows
    i_n=image.shape[1] #columns

    k_m=kernel.shape[0] #rows
    k_n=kernel.shape[1] #columns
    pad_h=(k_m-1)//2  
    
    pad_w=(k_n-1)//2

    
    if type=='zero':

       out_image=np.zeros(shape=(image.shape[0]+pad_h*2,image.shape[1]+pad_w*2))
       #print(out_image.shape)
       out_image[pad_h:i_m+pad_h,pad_w:i_n+pad_w] =np.copy(image)
    
    #change wrap padding corner
    if type=='wrap':
        out_image=np.zeros(shape=(image.shape[0]+pad_h*2,image.shape[1]+pad_w*2))
        out_image[0:pad_h,pad_w:i_n+pad_w]=image[i_m-pad_h:i_m,0:i_n]
        out_image[i_m+pad_h:out_image.shape[0],pad_w:i_n+pad_w]=image[0:pad_h,0:i_n]
        out_image[pad_h:i_m+pad_h,0:pad_w]=image[0:i_m,i_n-pad_w:i_n]
        out_image[pad_h:i_m+pad_h,i_n+pad_w:out_image.shape[1]]=image[0:i_m,0:pad_w]
        out_image[pad_h:i_m+pad_h,pad_w:i_n+pad_w] =np.copy(image)
        out_image[0,0]=np.copy(image[i_m-1,i_n-1])   #upper left corner
        out_image[i_m+pad_h,0]=np.copy(image[0,i_n-1]) #lower left corner
        out_image[0,i_n+pad_w]=np.copy(image[i_m-1,0]) #upper right corner
        out_image[i_m+pad_h:out_image.shape[0],i_n+pad_w:out_image.shape[1]]=np.copy(image[0:pad_h,0:pad_w]) #lower right corner
    
    if type=='copy':
        out_image=np.zeros(shape=(image.shape[0]+pad_h*2,image.shape[1]+pad_w*2))
        out_image[pad_h:i_m+pad_h,0:pad_w]=image[0:i_m,0:pad_w]
        out_image[pad_h:i_m+pad_h,i_n+pad_w:out_image.shape[1]]=image[0:i_m,i_n-pad_w:i_n]
        out_image[0:pad_h,pad_w:i_n+pad_w]=image[0:pad_h,0:i_n]
        out_image[i_m+pad_h:out_image.shape[0],pad_w:out_image.shape[1]-pad_w]=image[i_m-pad_h:i_m,0:i_n]
        out_image[pad_h:i_m+pad_h,pad_w:i_n+pad_w]=np.copy(image)
        out_image[0:pad_h,0:pad_w]=np.copy(image[0:pad_h,0:pad_w])   #upper left corner
        out_image[i_m+pad_h:out_image.shape[0],0:pad_w]=np.copy(image[i_m-pad_h:i_m,0:pad_w]) #lower left corner
        out_image[0:pad_h,i_n+pad_w:out_image.shape[1]]=np.copy(image[0:pad_h,0:pad_w]) #upper right corner
        out_image[i_m+pad_h:out_image.shape[0],i_n+pad_w:out_image.shape[1]]=np.copy(image[i_m-pad_h:i_m,i_n-pad_w:i_n]) #lower right corner

    if type=='reflect':
        out_image=np.zeros(shape=(image.shape[0]+pad_h*2,image.shape[1]+pad_w*2))
        out_image[pad_h:i_m+pad_h,0:pad_w]=np.flip(image[0:i_m,0:pad_w],0)
        out_image[pad_h:i_m+pad_h,i_n+pad_w:out_image.shape[1]]=np.flip(image[0:i_m,i_n-pad_w:i_n],0)
        out_image[0:pad_h,pad_w:i_n+pad_w]=np.flip(image[0:pad_h,0:i_n],0)
        out_image[i_m+pad_h:out_image.shape[0],pad_w:out_image.shape[1]-pad_w]=np.flip(image[i_m-pad_h:i_m,0:i_n],0)
        out_image[0,0]=np.copy(image[0+pad_h,0+pad_w]) #upper left corner
        out_image[out_image.shape[0]-1,0]=np.copy(image[i_m-pad_h,i_n-pad_w])#lower left corner
        out_image[0,out_image.shape[1]-1]=np.copy(image[0+pad_h,i_n-pad_w]) #upper right corner
        out_image[out_image.shape[0]-1,out_image.shape[1]-1]=np.copy(image[i_m-pad_h,i_n-pad_w])#lower right corner
        out_image[pad_h:i_m+pad_h,pad_w:i_n+pad_w] =np.copy(image)

    output_conv=np.zeros((i_m,i_n), dtype="uint8")

    for i in range(pad_h,i_m + pad_h):
        for j in range(pad_w,i_n+pad_w):

          roi =(out_image[i-pad_w:i+pad_w+1,j-pad_w:j+pad_w+1])*kernel
          k=roi.sum()
          output_conv[i-pad_w,j-pad_w]=k

    return output_conv

def rgb_conv(img,kernel,type):

    temp=np.zeros((img.shape[0],img.shape[1],3))
    temp[:,:,0]=conv2(img[:,:,0],kernel,type)
    temp[:,:,1]=conv2(img[:,:,1],kernel,type)
    temp[:,:,2]=conv2(img[:,:,2],kernel,type)

    return temp
    
def blend(laplaceA, laplaceB, maskPyr):

    maskPyr.pop()
    maskPyr = list(reversed(maskPyr))
    LS = []

    for la, lb, gm in zip(laplaceA, laplaceB, maskPyr):
        
        ls = la*gm + lb*(1.0 - gm)
        LS.append(ls)

    return LS

def reconstruct(laplacian_pyr, num_layers):
    #taking blended laplacian and reconstructing the original image
    
    laplacianTop = laplacian_pyr[0]
    sigma =  0.475 + (num_layers + 1)*sigmaStepSize

    for i in range(1, int(num_layers)):

        laplacianTop = cv.merge((sample(laplacianTop[:,:,0], 2), sample(laplacianTop[:,:,1], 2), sample(laplacianTop[:,:,2], 2)))
        laplacianTop = rgb_conv(laplacianTop, GaussianK(sigma), 'zero')
        laplacianTop = laplacianTop + laplacian_pyr[i]
        sigma = sigma - sigmaStepSize

    laplacianTop = rescale_intensity(laplacianTop, in_range=(0,255))
    
    return laplacianTop  

def GaussianK(sigma):
    
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size//2
    n = filter_size//2
    
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2
    
    return gaussian_filter

#sampling up/down controlled by factor ( use n for upsampling and 1/n for downsampling)
def sample(img,factor):

    w= math.ceil(factor*img.shape[0])
    h = math.ceil(factor*img.shape[1])
    new_img = np.zeros((w,h))

    for i in range(w):
        for j in range(h):
            temp = img[math.floor(i/factor),math.floor(j/factor)]
            new_img[i][j] = temp

    return new_img

def reshape(img):

    imgReshaped = np.zeros((img.shape[0], img.shape[1], 3))
    imgReshaped[:, :, 0] = img
    imgReshaped[:, :, 1] = img
    imgReshaped[:, :, 2] = img

    return imgReshaped

def main():

    imgSrc = (cv.resize((np.uint8(cv.imread('coentrao.png'))), (512, 512)))
    imgTrg = (cv.resize((np.uint8(cv.imread('cristianoReal.png'))), (512, 512)))
    
    cv.imwrite('imgA.png', imgSrc)
    cv.imwrite('imgB.png', imgTrg)
    
    
    max_layers = math.log2(min(imgSrc.shape[0], imgSrc.shape[1]))
    inputNumLayers = int(input("Enter desired level of pyramids: "))
    while(max_layers < inputNumLayers):
        prompt = "Invalid input. Enter less than {max_num} :".format(max_num = max_layers)
        inputNumLayers = int(input(prompt))

    #to get region of interest1
    plt.imshow(imgSrc)
    roi = RoiPoly(color='r')
    roi.display_roi()
    mask = (roi.get_mask(imgSrc[ :, :, 0]))
    mask = reshape(mask)
    imshow(mask)

    gPyrSrc, lPyrSrc = ComputePyr(imgSrc, int(inputNumLayers))
    gPyrTrg, lPyrTrg = ComputePyr(imgTrg, int(inputNumLayers))
    gPyrMask, lPyrMask = ComputePyr(mask, int(inputNumLayers))

    add_laplace = blend(lPyrSrc, lPyrTrg, gPyrMask)
    final = reconstruct(add_laplace, inputNumLayers)
    imshow(final.astype('float32'))

if __name__=='__main__':
    main()