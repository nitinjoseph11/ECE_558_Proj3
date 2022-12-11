import cv2 as cv
import numpy as np
import math
from skimage.exposure import rescale_intensity
from varname import nameof
from roipoly import RoiPoly
from matplotlib import pyplot as plt

def imshow(img):
    windowName = nameof(img)
    cv.imshow(windowName,img.astype('float32'))
    k = cv.waitKey(0)
    cv.destroyAllWindows
    return

def ComputePyr(img, noLayers):
    downsample = img.copy()
    gPyr = [downsample]
    #sigma = noLayers
    for i in range(noLayers):
        downsample = REDUCE(downsample)
        gPyr.append(downsample)
        #sigma = sigma - 1
    lPyr = EXPAND(gPyr)
    #gPyr.pop()
    return gPyr, lPyr

def REDUCE(img):
    #refer https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
    #gKernel = (1.0/256)*np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6], [4, 16, 24, 16, 4],[1, 4, 6, 4, 1]], dtype='uint8')
    gKernel = Gaussian()
    blur = convolveBW(img, gKernel) if len(img.shape) < 3 else convolveColor(img, gKernel) #blurring
    return blur[::2, ::2] #downsample

def EXPAND(gPyr):

    #gKernel = (1.0/256)*np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6], [4, 16, 24, 16, 4],[1, 4, 6, 4, 1]], dtype='uint8')
    
    numLevels = len(gPyr) -1
    lPyr = []
    lPyr.append(gPyr[numLevels - 1])
    #sigma = 1
    for i in range(numLevels - 1, 0, -1):
        gExpand = np.zeros((2*gPyr[i].shape[0], 2*gPyr[i].shape[1]))
        gExpand[::2, ::2] = gPyr[i]                          #upsample
        gKernel = Gaussian()
        gExpand = convolveBW(gExpand, gKernel) if len(gExpand.shape) < 3 else convolveColor(gExpand, gKernel)            #smoothen it
        laplacian = gPyr[i-1] - gExpand             #subtract from the previous gaussian pyramid
        lPyr.append(laplacian)
        #sigma = sigma + 1
    return lPyr

def upsample(img):
    imgUpsampled = np.zeros((2*img.shape[0], 2*img.shape[1]))
    imgUpsampled[::2, ::2] = img
    return imgUpsampled

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

    maskPyr.pop()
    maskPyr = normalize(maskPyr)
    maskPyr = list(reversed(maskPyr))
    LS = []
    for la, lb, gm in zip(laplaceA, laplaceB, maskPyr):
        ls = cv.add(la*gm, lb*(1.0 - gm))
        LS.append(ls)
    return LS

def reconstruct(laplacian_pyr, num_layers):
    #gKernel = (1.0/256)*np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6], [4, 16, 24, 16, 4],[1, 4, 6, 4, 1]])
    laplacianTop = laplacian_pyr[0]
    #sigma = 1
    for i in range(1, int(num_layers)):
        laplacianTop = upsample(laplacianTop)
        gKernel = Gaussian()
        laplacianTop = convolveBW(laplacianTop, gKernel) if len(laplacianTop.shape) < 3 else convolveColor(laplacianTop, gKernel)
        laplacianTop = laplacianTop + laplacian_pyr[i]
        #sigma = sigma + 1
    laplacianTop = rescale_intensity(laplacianTop, in_range=(0,255))
    return laplacianTop  

def normalize(pyrArray):
    returnList = list()
    for each in pyrArray:
        each = np.asarray(each)
        each = each/255.0
        returnList.append(each)
    return returnList

def returnShape(list):
    for each in list:
        print(each.shape)

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

def Gaussian(shape=(5,5),sigma=1):

    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def up_down(img,factor):

    w= math.ceil(factor*img.shape[0])
    h = math.ceil(factor*img.shape[1])
    #print(int(x1),int(x2))
    new_img = np.zeros((w,h))

    for i in range(w):
        for j in range(h):
            temp = img[math.floor(i/factor),math.floor(j/factor)]
            new_img[i][j] = temp
    return new_img
    
def main():
    imgSrc = (cv.resize((cv.imread('LH.png')), (512, 512)))
    imgTrg = (cv.resize((cv.imread('MS.png')), (512, 512)))
    imgSrc = cv.cvtColor(imgSrc, cv.COLOR_BGR2GRAY)
    imgTrg = cv.cvtColor(imgTrg, cv.COLOR_BGR2GRAY)
    num_layers = math.log2(min(imgSrc.shape[0], imgSrc.shape[1]))
    plt.imshow(imgSrc)
    roi = RoiPoly(color='r')
    roi.display_roi()
    mask = (roi.get_mask(imgSrc[:,:]))
    mask = (mask.astype(int))*255
    #print('mask.shape', mask.shape)
    gPyrSrc, lPyrSrc = ComputePyr(imgSrc, int(num_layers))
    gPyrTrg, lPyrTrg = ComputePyr(imgTrg, int(num_layers))
    mask = (np.reshape(mask, (512, 512))).astype('uint8')
    imshow(mask)
    gPyrMask, lPyrMask = ComputePyr(mask, int(num_layers))

    add_laplace = blend(lPyrSrc, lPyrTrg, gPyrMask)
    final = reconstruct(add_laplace, num_layers)
    imshow(final)
    print(final.shape) 
    
    
if __name__=='__main__':
    main()