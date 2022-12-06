import cv2 as cv
import numpy as np
import math
from skimage.exposure import rescale_intensity
from varname import nameof
from roipoly import RoiPoly
from matplotlib import pyplot as plt

def imshow(img):
    windowName = nameof(img)
    cv.imshow(windowName,img)
    k = cv.waitKey(0)
    cv.destroyAllWindows
    return

def ComputePyr(img, noLayers):
    downsample = img.copy()
    gPyr = [downsample]
    for i in range(noLayers):
        downsample = REDUCE(downsample)
        gPyr.append(downsample)
    lPyr = EXPAND(gPyr)
    gPyr.pop()
    return gPyr, lPyr

def REDUCE(img):
    #refer https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
    gKernel = (1.0/256)*np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6], [4, 16, 24, 16, 4],[1, 4, 6, 4, 1]])
    # print('img.shape inside REDUCE', img.shape)
    blur = convolveBW(img, gKernel) if len(img.shape) < 3 else convolveColor(img, gKernel) #blurring
    return blur[::2, ::2] #downsample

def EXPAND(gPyr):
    gKernel = (1.0/256)*np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6], [4, 16, 24, 16, 4],[1, 4, 6, 4, 1]])
    numLevels = len(gPyr) - 1
    lPyr = []
    for i in range(numLevels, 0, -1):
        gExpand = np.zeros((2*gPyr[i].shape[0], 2*gPyr[i].shape[1]))
        gExpand[::2, ::2] = gPyr[i]                          #upsample
        gExpand = convolveBW(gExpand, 4*gKernel) if len(gExpand.shape) < 3 else convolveColor(gExpand, 4*gKernel)            #smoothen it
        laplacian = np.subtract(gPyr[i-1], gExpand)             #subtract from the previous gaussian pyramid
        lPyr.append(laplacian)
    return lPyr

def padImage(img, kernel):
    paddingWidthx = paddingWidthy = top = bottom = left = right = math.ceil((kernel.shape[0] - 1)/2) #because m and n are equal for box
    paddedImg = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, None, value = 0)
    return paddedImg, paddingWidthx, paddingWidthy

def conv2d(f,w):
    fp, padWidthx, padWidthy = padImage(f,w)
    convolvedOutput = np.zeros((fp.shape[0] - 2*padWidthx,fp.shape[1] - 2*padWidthy),dtype="float32")
    for row in np.arange(padWidthx, fp.shape[0] - padWidthx):
        for col in np.arange(padWidthy, fp.shape[1] - padWidthy):
            roi = fp[row-padWidthx:row+padWidthx+1, col-padWidthy:col+padWidthy+1] 
            conv = np.sum(np.multiply(roi,w))
            convolvedOutput[row - padWidthx, col - padWidthy] = conv
    convolvedOutput = rescale_intensity(convolvedOutput,in_range =(0,255))
    convolvedOutput = (convolvedOutput*255).astype("uint8")
    return convolvedOutput

def convolveColor(f,w):
    try:
        fB, fG,fR = f[:,:,0], f[:,:,1], f[:,:,2]
        convB = conv2d(fB,w)
        convG = conv2d(fG,w)
        convR = conv2d(fR,w)
        return np.dstack((convB,convG,convR))
    except:
        conv = conv2d(f, w)
        return conv

def convolveBW(f,w):
    convBW = conv2d(f,w)
    return convBW

def blend(laplaceA, laplaceB, maskPyr):
    maskPyr = list(reversed(maskPyr))
    LS = []
    for la, lb, gm in zip(laplaceA, laplaceB, maskPyr):
        ls = la*gm + lb*(255.0 - gm)
        LS.append(ls)
    return LS

def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        #laplacian_expanded = cv.pyrUp(laplacian_top, dstsize=size)
        laplacian_expanded = np.zeros((2*laplacian_top.shape[0], 2*laplacian_top.shape[1]))
        laplacian_expanded[::2, ::2] = laplacian_top
        laplacian_top = cv.add(laplacian_pyr[i+1], laplacian_expanded)
        #laplacian_lst.append(laplacian_top)
    return laplacian_top  

def main():
    imgSrc = (cv.resize((cv.imread('greyGirl.png')), (512, 512)))
    imgTrg = (cv.resize((cv.imread('monaLisaBW.jpg')), (512, 512)))
    imgSrc = cv.cvtColor(imgSrc, cv.COLOR_BGR2GRAY)
    imgTrg = cv.cvtColor(imgTrg, cv.COLOR_BGR2GRAY)
    # print(imgSrc.shape, imgTrg.shape)
    # sampling_rate ^ (num_layers) <= min(m,n)
    num_layers = math.log2(min(imgSrc.shape[0], imgSrc.shape[1]))
    plt.imshow(imgSrc)
    roi = RoiPoly(color='r')
    roi.display_roi()
    mask = (roi.get_mask(imgSrc[:,:]))
    mask = (mask.astype(int))*255
    #print('mask.shape', mask.shape)
    gPyrSrc, lPyrSrc = ComputePyr(imgSrc, int(num_layers))
    gPyrTrg, lPyrTrg = ComputePyr(imgTrg, int(num_layers))
    # plt.imshow(mask, interpolation='nearest')
    # plt.show()
    mask = (np.reshape(mask, (512, 512))).astype('uint8')
    imshow(mask)
    #mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    gPyrMask, lPyrMask = ComputePyr(mask, int(num_layers))
    add_laplace = blend(lPyrSrc, lPyrTrg, gPyrMask)
    final = reconstruct(add_laplace)
    imshow(final)
    
    
if __name__=='__main__':
    main()