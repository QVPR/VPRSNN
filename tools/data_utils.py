'''
MIT License

Copyright (c) 2023 Somayeh Hussaini, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''




import os

import cv2
import numpy as np


def loadImg(imPath):

    # read and convert image from BGR to RGB 
    im = cv2.imread(imPath)[:,:,::-1]

    return im


def get_patches2D(image, patch_size):

    if patch_size[0] % 2 == 0: 
        nrows = image.shape[0] - patch_size[0] + 2
        ncols = image.shape[1] - patch_size[1] + 2
    else:
        nrows = image.shape[0] - patch_size[0] + 1
        ncols = image.shape[1] - patch_size[1] + 1
    return np.lib.stride_tricks.as_strided(image, patch_size + (nrows, ncols), image.strides + image.strides).reshape(patch_size[0]*patch_size[1],-1)


def patch_normalise_pad(image, patch_size):

    patch_size = (patch_size, patch_size)
    patch_half_size = [int((p-1)/2) for p in patch_size ]

    image_pad = np.pad(np.float64(image), patch_half_size, 'constant', constant_values=np.nan)

    nrows = image.shape[0]
    ncols = image.shape[1]
    patches = get_patches2D(image_pad, patch_size)
    mus = np.nanmean(patches, 0)
    stds = np.nanstd(patches, 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        out = (image - mus.reshape(nrows, ncols)) / stds.reshape(nrows, ncols)

    out[np.isnan(out)] = 0.0
    out[out < -1.0] = -1.0
    out[out > 1.0] = 1.0
    return out


def processImage(img, imWidth, imHeight, num_patches):

    img = cv2.resize(img,(imWidth, imHeight))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    im_norm = patch_normalise_pad(img, num_patches) 

    # Scale element values to be between 0 - 255
    img = np.uint8(255.0 * (1 + im_norm) / 2.0)

    return img


def get_filtered_imageNames(filtered_imageNames_path):

    assert os.path.isfile(filtered_imageNames_path), "The file path {} is not a valid".format(filtered_imageNames_path)

    with open(filtered_imageNames_path) as f:
        content = f.read().splitlines()
    
    return content


def filter_frames(frames, labels, paths_used, type, shuffled_indices, imgLists, offset_after_skip, num_labels, num_test_labels):
    
    # shuffle data 
    paths_used[:len(shuffled_indices)] = paths_used[shuffled_indices]
    frames[:len(shuffled_indices)] = frames[shuffled_indices]
    
    if len(imgLists) == 2:
        paths_used[len(shuffled_indices):] = paths_used[len(shuffled_indices):][shuffled_indices]
        frames[len(shuffled_indices):] = frames[len(shuffled_indices):][shuffled_indices]
    
    if type == "train":
        
        # extract training data 
        training_indices = np.arange(offset_after_skip, offset_after_skip+num_labels)
        
        paths_used_filtered = paths_used[training_indices]
        frames_filtered = frames[training_indices]
        labels_filtered = labels[training_indices]
        
        if len(imgLists) == 2:
            paths_used_filtered = np.append(paths_used_filtered, paths_used[len(shuffled_indices):][training_indices])
            frames_filtered = np.append(frames_filtered, frames[len(shuffled_indices):][training_indices], axis=0)
            labels_filtered = np.append(labels_filtered, labels[len(shuffled_indices):][training_indices])
            
        paths_used = paths_used_filtered
        frames = frames_filtered
        labels = labels_filtered
    
    elif type == "test":
        
        paths_used = paths_used[offset_after_skip:offset_after_skip+num_test_labels]
        frames = frames[offset_after_skip:offset_after_skip+num_test_labels]
        labels = labels[offset_after_skip:offset_after_skip+num_test_labels]
        
    return frames, labels, paths_used 


def processImageDataset(path, type, imWidth, imHeight, num_patches=7, num_labels=25, num_test_labels=2700, num_query_imgs=3300, skip=0, offset_after_skip=0, shuffled_indices=[]):

    print("Computing features for image names in path(s): {} ...\n".format(path))

    dataset_path = './../data'
    imgLists = []
    
    for p in path: 
        imgList = get_filtered_imageNames(p)  
        imgLists.append(imgList)

    frames = []
    paths_used = [] 
    labels = []

    for imgList in imgLists: 

        ii = 0  # keep count of number of images
        kk = 0  # keep track of image indices, considering offset after skip

        for i, imPath in enumerate(imgList):
            
            if (ii == num_query_imgs):
                break 

            if ".jpg" not in imPath and ".png" not in imPath:
                continue 
            
            if skip != 0 and i % skip != 0:  
                continue
            
            if not shuffled_indices.any() and (offset_after_skip > 0 and kk < offset_after_skip):
                kk += 1
                continue
            
            if shuffled_indices.any() and kk not in shuffled_indices:
                kk += 1
                continue
            
            frame = loadImg(os.path.join(dataset_path, imPath))

            frame = processImage(frame, imWidth, imHeight, num_patches)  
            frames.append(frame)

            labels.append(ii)

            path_id = i

            paths_used.append(path_id)

            ii += 1
            kk += 1 
            
    if shuffled_indices.any(): 
        # All data is loaded, now extract the relevant frames and labels
        frames, labels, paths_used = filter_frames(np.array(frames), np.array(labels), np.array(paths_used), type, shuffled_indices, imgLists, offset_after_skip, num_labels, num_test_labels)
    
    print("indices used in {}:\n{}".format(type, paths_used))
    print("labels used in {}:\n{}".format(type, labels))

    y = np.array([ [labels[i]] for i in range(len(labels)) ])
    data = {'x': np.array(frames), 'y': y, 'rows': imWidth, 'cols': imHeight}

    return data


def get_train_test_imagenames_path(dataset, folder_id):

    if dataset == 'nordland':
        if folder_id == "NRD_SFS":
            train_data_path = [f'./dataset_imagenames/{dataset}_imageNames_spring.txt', f'./dataset_imagenames/{dataset}_imageNames_fall.txt']  
            test_data_path =  [f'./dataset_imagenames/{dataset}_imageNames_summer.txt']
        elif folder_id == "NRD_SFW":
            train_data_path = [f'./dataset_imagenames/{dataset}_imageNames_spring.txt', f'./dataset_imagenames/{dataset}_imageNames_fall.txt']  
            test_data_path =  [f'./dataset_imagenames/{dataset}_imageNames_winter.txt']
            
    elif dataset == 'ORC':
        train_data_path = [f'./dataset_imagenames/{dataset}_imageNames_Sun.txt', f'./dataset_imagenames/{dataset}_imageNames_Rain.txt']
        test_data_path =  [f'./dataset_imagenames/{dataset}_imageNames_Dusk.txt']
        
    elif dataset == 'SFU-Mountain':
        train_data_path = [f'./dataset_imagenames/{dataset}_imageNames_dry.txt']
        test_data_path = [f'./dataset_imagenames/{dataset}_imageNames_dusk.txt']

    elif dataset == 'Synthia-NightToFall':
        train_data_path = [f'./dataset_imagenames/{dataset}_imageNames_ref.txt']
        test_data_path = [f'./dataset_imagenames/{dataset}_imageNames_query.txt']

    elif dataset == 'St-Lucia':
        train_data_path = [f'./dataset_imagenames/{dataset}_imageNames_190809_0845.txt']
        test_data_path = [f'./dataset_imagenames/{dataset}_imageNames_180809_1545.txt']

    return train_data_path, test_data_path


