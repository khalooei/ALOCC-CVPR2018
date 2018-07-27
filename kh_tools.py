import os
import datetime
import json
from PIL import Image
import numpy as np
from glob import glob
from PIL import Image, ImageDraw
from scipy import ndimage, misc
import scipy.misc
import imageio

# http://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise.html
from skimage.util import random_noise

'''
IMAGE PROCESSING
- read_dataset_image_path
- read_dataset_images
- read_lst_images
- read_image
- get_noisy_data
'''
def get_noisy_data(data):
    lst_noisy = []
    sigma = 0.155
    for image in data:
        noisy = random_noise(image, var=sigma ** 2)
        lst_noisy.append(noisy)
    return np.array(lst_noisy)

def read_dataset_image_path(s_dataset_url, n_number_count=None):
    lst_dir_inner_images_path = []
    for s_dir_path in glob(os.path.join(s_dataset_url, '*')):
        for s_image_path in glob(os.path.join(s_dir_path, '*')):
            lst_dir_inner_images_path.append(s_image_path)
            if n_number_count is not None:
                if (len(lst_dir_inner_images_path) >= n_number_count):
                    return np.array(lst_dir_inner_images_path)

    return lst_dir_inner_images_path



def read_image_w_noise(s_image_path):
    tmp_image = read_image(s_image_path)
    sigma = 0.155
    noisy = random_noise(tmp_image, var=sigma ** 2)
    # image = scipy.misc.imresize(tmp_image, nd_img_size)
    return np.array(noisy)

def read_lst_images_w_noise2(lst_images_path,nd_patch_size, n_patch_step):
    lst_images = []
    for image_path in lst_images_path:
        lst_images.append(read_image_w_noise(image_path))
    return np.array(lst_images)

def read_lst_images_w_noise(lst_images_path,nd_patch_size, n_patch_step):
    lst_slices = []
    lst_location = []
    for image_path in lst_images_path:
        tmp_img = read_image_w_noise(image_path)
        tmp_slices,tmp_location_slice = get_image_patches([tmp_img], nd_patch_size, n_patch_step)
        lst_slices.extend(tmp_slices)
        lst_location.extend(tmp_location_slice)
    return np.array(lst_slices),lst_location

def read_lst_images(lst_images_path,nd_patch_size, n_patch_step, b_work_on_patch=True):
    if b_work_on_patch:
        lst_slices = []
        lst_location = []
        for image_path in lst_images_path:
            tmp_img = read_image(image_path)
            tmp_slices,tmp_location_slice = get_image_patches([tmp_img], nd_patch_size, n_patch_step)
            lst_slices.extend(tmp_slices)
            lst_location.extend(tmp_location_slice)
        return lst_slices,lst_location
    else:
        lst_images = []
        for image_path in lst_images_path:
            lst_images.append(read_image(image_path))
        return np.array(lst_images)

def read_dataset_images(s_dataset_url , nd_img_size ,n_number_count):
    lst_images = []
    for s_dir_path in glob(os.path.join(s_dataset_url, '*')):
        for s_image_path in glob(os.path.join(s_dir_path, '*')):
            lst_images.append(read_image(s_image_path,nd_img_size))
            if n_number_count is not None:
                if (len(lst_images)>= n_number_count):
                    return np.array(lst_images)

    return np.array(lst_images)

def read_image(s_image_path):
    tmp_image = scipy.misc.imread(s_image_path)[100:240,0:360]/127.5 -1.
    #sigma = 0.155
    #noisy = random_noise(tmp_image, var=sigma ** 2)
    # image = scipy.misc.imresize(tmp_image, nd_img_size)
    return np.array(tmp_image)


def get_patch_video(lst_images, nd_patch_size, nd_stride, n_depth):
    lst_video_slice = []
    lst_video_location = []
    n_video_numbers = len(lst_images) // n_depth

    flag = True
    n_video_slices_number = 0

    for i in range(0, n_video_numbers):
        tmp_video = read_lst_images(lst_images[i * n_depth:((i + 1) * n_depth)])
        lst_tmp_video, lst_tmp_location = get_image_patches(tmp_video, nd_patch_size, nd_stride)

        if flag:
            flag = False
            n_video_slices_number = len(lst_tmp_video)

        lst_video_slice.extend(lst_tmp_video)
        lst_video_location.extend(lst_tmp_location)

    print('video patches is ready ({} patches)'.format(len(lst_video_slice)))

    return np.array(lst_video_slice), lst_video_location


def get_image_patches(image_src, nd_patch_size, nd_stride):
    image_src = np.array(image_src)

    lst_patches = []
    lst_locations = []

    n_stride_h = nd_stride[0]
    n_stride_w = nd_stride[1]

    tmp_frame = image_src[0].shape
    n_frame_h = tmp_frame[0]
    n_frame_w = tmp_frame[1]

    # for i in range(0,n_frame_h,n_stride_h):
    # np.array(lst_patches[10])[0,:,:]
    flag_permission_h = flag_permission_w = True
    i = 0
    while i < n_frame_h and flag_permission_h:
        flag_permission_w = True
        start_h = i
        end_h = i + nd_patch_size[0]
        if end_h > n_frame_h:
            end_h = n_frame_h
            start_h = n_frame_h - nd_patch_size[0]
            # break
        # for j in range(0,n_frame_w,n_stride_w):
        j = 0
        while j < n_frame_w and flag_permission_w:
            start_w = j
            end_w = j + nd_patch_size[1]
            if end_w > n_frame_w:
                end_w = n_frame_w
                start_w = n_frame_w - nd_patch_size[1]
                # break

            # print(start_w,end_w,'**',start_h,end_h)

            tmp_slices = np.array(image_src[:, start_h:end_h, start_w:end_w])
            lst_patches.append(tmp_slices)
            lst_locations.append([start_h, start_w])

            j += n_stride_w
            if j > n_frame_w:
                flag_permission_w = False
                j = n_frame_w - nd_patch_size[1]

        i += n_stride_h
        if i > n_frame_h:
            flag_permission_h = False
            i = n_frame_h - nd_patch_size[0]

    return np.array(lst_patches), lst_locations

def kh_isDirExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('path ',path,' is created')
    return

def kh_crop(img,nStartX,nEndX,nStartY,nEndY):
    return img[nStartY:nEndY,nStartX:nEndX]

def kh_extractPatches(sImg,nStride=1,ndSliceSize=(10,10),bSaveImages=False):
    i = 0
    j = 0
    imgArray = np.zeros([Image.open(sImg[0]).size[1],Image.open(sImg[0]).size[0], 3])

    while i<len(sImg):
        # read Images
        imgTmp1 = Image.open(sImg[i])
        imgTmp2 = Image.open(sImg[i+1])

        #Image to Numpy array
        imgArray1 = np.array(imgTmp1)
        imgArray2 = np.array(imgTmp1)

        A = imgArray1
        A = (A - np.mean(A)) / np.std(A)
        imgArray1 = A

        A = imgArray2
        A = (A - np.mean(A)) / np.std(A)
        imgArray2 = A

        imgArray[:, :, j] =np.add(imgArray1,imgArray2)/2

        i=i+2
        j=j+1



    #=========================================================
    nImgArrayH = imgArray.shape[0]
    nImgArrayW =  imgArray.shape[1]

    best_rg = imgArray[100:nImgArrayH-14,0:nImgArrayW]
    #best_rg = imgArray[0:nImgArrayH, 0:nImgArrayW]
    ndMainSize=(best_rg.shape[0],best_rg.shape[1])

    ndSliceSizeWidth = ndSliceSize[0]
    ndSliceSizeHeight =  ndSliceSize[1]

    # Copy master
    path = os.path.dirname(sImg[0])
    base = os.path.basename(sImg[0])

    # slice the image to 1000 x 1000 tiles
    slice_size = ndSliceSizeWidth
    lst_fNamesTmp=[]
    lst_Patches=[]
    beforeViewedX = []
    beforeViewedY = []
    for y in range(0,nImgArrayH-ndSliceSizeHeight+1, nStride):
        for x in range(0, nImgArrayW-ndSliceSizeWidth+1, nStride):
            #fname = os.path.join(path, sPrefixOutput+"/p-%d-%d-%s" % (x, y, base))
            #basePosition = "%s--[%d,%d]--(%d,%d)" % (sFileAddress,ndSliceSizeWidth,ndSliceSizeHeight,x, y)

            #print("Creating tile:" + basePosition)

            minX=x
            minY=y
            if ((x + slice_size) >= nImgArrayW):
                minX = x - slice_size -1
            else:
                minX = x

            if ((y + slice_size) >= nImgArrayH):
                minY = y - slice_size -1
            else:
                minY = y

            mx = min(x + slice_size, nImgArrayW)
            my = min(y + slice_size, nImgArrayH)

            if(mx or x) > nImgArrayW   and (my or y)>nImgArrayH:
                continue

            sSaveBasePatchImg='./'#+'/' + base



            basePosition = "(%d,%d)" % (minX, minY)
            saveAddress = sSaveBasePatchImg + '/' +path[(len(path)-8):len(path)]+'_'+base[0:3]+'_'+basePosition


            #buffer = Image.new("RGB", [slice_size, slice_size], (255, 255, 255))
            #buffer = Image.new("YCbCr", [slice_size, slice_size])
            #tile = imgTmp.crop((minX, minY, mx, my))
            crp = kh_crop(imgArray,minX,mx,minY,my)
            tile = np.resize(crp,[slice_size,slice_size,3])

            # tmpArr=np.array(tile)
            # tile = Image.fromarray(tmpArr)
            #buffer.paste(tile.resize(ndSliceSize), (0, 0))

            if bSaveImages:
                kh_isDirExist(sSaveBasePatchImg)
                #buffer.save(saveAddress, "JPEG")
                npTile = np.array(tile.resize(ndSliceSize));
                scipy.misc.imsave(saveAddress+'.jpg', npTile)

            if True:#basePosition not in lst_fNamesTmp:
                lst_fNamesTmp.append(basePosition)
                # Image to Numpy array
                #imgBuffer = np.array(buffer)
                #imgBuffer = np.expand_dims(np.array(tile.resize(ndSliceSize)), axis=-1)
                #expand_tile = np.expand_dims(tile,-1)
                #buffer = Image.new("RGB", [slice_size, slice_size], (100, 10, 100))
                #buffer.paste(Image.fromarray(tile))
                # img = np.zeros([ndSliceSizeWidth, ndSliceSizeHeight, 3])
                # img[:, :, 0] = tile
                # img[:, :, 1] = tile
                # img[:, :, 2] = tile
                lst_Patches.append(tile)
                #print('add => ', saveAddress)
            else:
                print('it is copy => ',basePosition)

    if bSaveImages:
        #buffer = Image.new("RGB", [ndMainSize[1], ndMainSize[0]], (255, 255, 255))
        #buffer.paste(imgTmp, (0, 0))
        npImgTmp = np.array(tile);
        scipy.misc.imsave(sSaveBasePatchImg+'/main'+base + '.jpg', npImgTmp)

    print(sImg,' => is finished')
    return lst_Patches,lst_fNamesTmp



def kh_extractPatchesOne(sImg,nStride=1,ndSliceSize=(10,10),bSaveImages=False):

    # read Images
    imgTmp = Image.open(sImg)

    #Image to Numpy array
    imgArray = np.array(imgTmp)

    #=========================================================
    nImgArrayH = imgArray.shape[0]
    nImgArrayW =  imgArray.shape[1]

    best_rg = imgArray[100:nImgArrayH-14,0:nImgArrayW]
    #best_rg = imgArray[0:nImgArrayH, 0:nImgArrayW]
    ndMainSize=(best_rg.shape[0],best_rg.shape[1])

    ndSliceSizeWidth = ndSliceSize[0]
    ndSliceSizeHeight =  ndSliceSize[1]

    # Copy master
    path = os.path.dirname(sImg)
    base = os.path.basename(sImg)

    # slice the image to 1000 x 1000 tiles
    slice_size = ndSliceSizeWidth
    lst_fNamesTmp=[]
    lst_Patches=[]
    beforeViewedX = []
    beforeViewedY = []
    for y in range(0,nImgArrayH-ndSliceSizeHeight+1, nStride):
        for x in range(0, nImgArrayW-ndSliceSizeWidth+1, nStride):
            #fname = os.path.join(path, sPrefixOutput+"/p-%d-%d-%s" % (x, y, base))
            #basePosition = "%s--[%d,%d]--(%d,%d)" % (sFileAddress,ndSliceSizeWidth,ndSliceSizeHeight,x, y)

            #print("Creating tile:" + basePosition)

            minX=x
            minY=y
            if ((x + slice_size) >= nImgArrayW):
                minX = x - slice_size -1
            else:
                minX = x

            if ((y + slice_size) >= nImgArrayH):
                minY = y - slice_size -1
            else:
                minY = y

            mx = min(x + slice_size, nImgArrayW)
            my = min(y + slice_size, nImgArrayH)

            if(mx or x) > nImgArrayW   and (my or y)>nImgArrayH:
                continue

            sSaveBasePatchImg='./'#+'/' + base



            basePosition = "(%d,%d)" % (minX, minY)
            saveAddress = sSaveBasePatchImg + '/' +path[(len(path)-8):len(path)]+'_'+base[0:3]+'_'+basePosition


            #buffer = Image.new("RGB", [slice_size, slice_size], (255, 255, 255))
            #buffer = Image.new("YCbCr", [slice_size, slice_size])
            #tile = imgTmp.crop((minX, minY, mx, my))
            crp = kh_crop(imgArray,minX,mx,minY,my)
            tile = np.resize(crp,[slice_size,slice_size])

            # tmpArr=np.array(tile)
            # tile = Image.fromarray(tmpArr)
            #buffer.paste(tile.resize(ndSliceSize), (0, 0))

            if bSaveImages:
                kh_isDirExist(sSaveBasePatchImg)
                #buffer.save(saveAddress, "JPEG")
                npTile = np.array(tile.resize(ndSliceSize));
                scipy.misc.imsave(saveAddress+'.jpg', npTile)

            if True:#basePosition not in lst_fNamesTmp:
                lst_fNamesTmp.append(basePosition)
                # Image to Numpy array
                #imgBuffer = np.array(buffer)
                #imgBuffer = np.expand_dims(np.array(tile.resize(ndSliceSize)), axis=-1)
                #expand_tile = np.expand_dims(tile,-1)
                #buffer = Image.new("RGB", [slice_size, slice_size], (100, 10, 100))
                #buffer.paste(Image.fromarray(tile))
                img = np.zeros([ndSliceSizeWidth, ndSliceSizeHeight, 3])
                img[:, :, 0] = tile
                img[:, :, 1] = tile
                img[:, :, 2] = tile
                lst_Patches.append(img)
                #print('add => ', saveAddress)
            else:
                print('it is copy => ',basePosition)

    if bSaveImages:
        #buffer = Image.new("RGB", [ndMainSize[1], ndMainSize[0]], (255, 255, 255))
        #buffer.paste(imgTmp, (0, 0))
        npImgTmp = np.array(tile);
        scipy.misc.imsave(sSaveBasePatchImg+'/main'+base + '.jpg', npImgTmp)

    print(sImg,' => is finished')
    return lst_Patches,lst_fNamesTmp

def kh_getSliceImages_simple(sBaseImageFiles='',ndSliceSize=(10,10),nStride=1,bSaveImages=False):
    lst_pics=[]
    lst_names=[]

    for sImagePath in sBaseImageFiles:
        lst_picTmp,lst_nameTmp = kh_extractPatchesOne(sImagePath,nStride=nStride,ndSliceSize=ndSliceSize,bSaveImages=bSaveImages)

        lst_pics= lst_pics+lst_picTmp
        lst_names=lst_names+lst_nameTmp

    return lst_pics,lst_names


def kh_getSliceImages(sBaseImageFiles='',ndSliceSize=(10,10),nStride=1,bSaveImages=False):
    lst_pics=[]
    lst_names=[]

    i=0
    while (i+6)<len(sBaseImageFiles):
        sImagePath= []
        sImagePath.append(sBaseImageFiles[i])
        sImagePath.append(sBaseImageFiles[i + 1])
        sImagePath.append(sBaseImageFiles[i + 2])
        sImagePath.append(sBaseImageFiles[i + 3])
        sImagePath.append(sBaseImageFiles[i + 4])
        sImagePath.append(sBaseImageFiles[i + 5])
        lst_picTmp,lst_nameTmp = kh_extractPatches(sImagePath,nStride=nStride,ndSliceSize=ndSliceSize,bSaveImages=bSaveImages)

        lst_pics= lst_pics+lst_picTmp
        lst_names=lst_names+lst_nameTmp

        i = i + 1

    return lst_pics,lst_names

def kh_getImages(sBaseImageFiles='',bGetSlice=True,ndSliceSize=(10,10),nStride=1,bSaveImages=False):
    if bGetSlice:
        return kh_getSliceImages(sBaseImageFiles=sBaseImageFiles,
                          ndSliceSize=ndSliceSize,
                          nStride=nStride,
                          bSaveImages=bSaveImages)
    return ('','')

