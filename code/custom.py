# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:43:15 2019

@author: dell
"""

import numpy as np
import util
from matplotlib import pyplot as plt
import skimage.io
from skimage import transform
import numpy as np
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import visual_words
import skimage.io

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    
    ''' Check image to make it a floating point with range[0,1] '''
    if image.any()>1:
        image = image.astype('float')/255
    
    ''' Convert to 3 channels if not '''
    if len(image.shape) == 2:
        image = np.tile(image[:, np.newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]
          
    ''' Convert image into lab color space '''    
    image = skimage.color.rgb2lab(image)
    
    ''' Apply filters '''
    scales = [1, 2, 4, 8, 8*np.sqrt(2)]
    for scale in range(len(scales)):
        for c in range(3):
            #img = skimage.transform.resize(image, (int(ss[0]/scales[i]),int(ss[1]/scales[i])),anti_aliasing=True)
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[scale])
            if scale == 0 and c == 0:
                filter_responses = img[:,:,np.newaxis]
            else:
                filter_responses = np.concatenate((filter_responses,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[scale])
            filter_responses = np.concatenate((filter_responses,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[scale],order=[0,1])
            filter_responses = np.concatenate((filter_responses,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[scale],order=[1,0])
            filter_responses = np.concatenate((filter_responses,img[:,:,np.newaxis]),axis=2)


    return filter_responses
    

def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    # ----- TODO -----
    filter_responses = extract_filter_responses(image)
    H, W = image.shape[0], image.shape[1]
    filter_responses = filter_responses.reshape(H*W, -1)
    
    dists = scipy.spatial.distance.cdist(filter_responses, dictionary)
    wordmap = np.argmin(dists, axis = 1).reshape(H,W)

    return wordmap
    pass
    


def compute_dictionary(num_workers):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    
    train_data = np.load("../data/train_data.npz")
    # ----- TODO -----
    
    ''' iterate through the paths to read images '''
    i = 0 # index of training image
    alpha = 50 # alpha between 50 and 500
    K = 100 # K between 100 and 200  
    
    for image_path in train_data.f.files:
        #print(i)
        args = i, alpha, image_path 
        compute_dictionary_one_image(args)
        
        current_compute = np.load(str(i)+".npy")
        if i==0:
            filter_responses = current_compute
        else:
            filter_responses = np.append(filter_responses, current_compute, axis=0)
     
        i+=1
    
    np.save("filter_responses.npy",filter_responses)     
    #filter_responses =  np.load("filter_responses.npy")
    kmeans = sklearn.cluster.KMeans(n_clusters = K,random_state=0).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    
    np.save("dictionary.npy",dictionary)  


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''

    i,alpha,image_path = args
    # ----- TODO -----
    image = skimage.io.imread('../data/'+image_path)
    image = image.astype('float')/255
    image = transform.downscale_local_mean(image, (3,3,1))
    filter_response = extract_filter_responses(image)
    
    ''' get random pixels '''
    #alpha_response = (np.random.permutation(filter_response))
    alpha_response = np.zeros(shape = (alpha,filter_response.shape[2]))
    for j in range(alpha):
        row = np.random.randint(0,filter_response.shape[0])
        col = np.random.randint(0,filter_response.shape[1])
        alpha_response[j]=filter_response[row][col]

    np.save(str(i)+".npy",alpha_response)




def build_recognition_system(num_workers):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,K*(4^layer_num-1)/3)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----
    SPM_layer_num = 4
    
    labels = train_data.f.labels
    
    N = len(labels)
    dict_size = dictionary.shape[0] 
    features = np.zeros((N, dict_size*((4**SPM_layer_num-1)//3)))
    files = train_data.f.files
    

    i=0
    for image_path in files:
        feature = get_image_feature(image_path,dictionary,SPM_layer_num,dict_size)
        features[i] = feature
        i+=1
        #print(i)
    
    trained_system = features, labels, dictionary, SPM_layer_num
    np.save("trained_system.npz",trained_system)
    pass

def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz.npy")
    # ----- TODO -----
    features, train_labels, dictionary, SPM_layer_num = trained_system
    test_labels = test_data.f.labels      
    files = test_data.f.files
    
    accurate = 0
    count=0
    conf = [[0 for i in range(8)] for i in range(8)]
    
    for i in range(len(files)):

        word_hist = get_image_feature(files[i],dictionary,SPM_layer_num,dictionary.shape[0])
        distance = distance_to_set(word_hist, features)
        label = train_labels[np.argmax(distance)]
        
        
        conf[test_labels[i]][label]+=1
        count+=1
        if test_labels[i]==label:
            accurate+=1
        accuracy = accurate/count
        #print("accurate:",test_labels[i],"predict:", label, "accuracy:",accuracy)    
    conf = np.array(conf)
    return conf, accuracy

def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    * histograms: numpy.ndarray of shape (N,K*(4^layer_num-1)/3)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    sim = np.minimum(word_hist, histograms)
    return np.sum(sim, axis=1)


def get_image_feature(file_path,dictionary,layer_num,K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    # ----- TODO -----
    image = skimage.io.imread('../data/'+file_path)
    image = image.astype('float')/255
    image = transform.downscale_local_mean(image, (3,3,1))
    wordmap = visual_words.get_visual_words(image, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
    return feature


def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    # ----- TODO -----   
    hist,bins = np.histogram(wordmap, bins=list(range(dict_size + 1)), density=True)    
              
    return hist



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    
    # ----- TODO -----
    H = wordmap.shape[0]
    W = wordmap.shape[1]
    
    # compute the finest, L=2
    step_h = H // 4
    step_w = W // 4
    
    hist_all = np.zeros(dict_size*((4**layer_num-1)//3))
    
    i,h,w = 0,0,0
    
    for h in range(0, step_h*4, step_h):  
        for w in range(0, step_w*4, step_w):   
            sub_wordmap = wordmap[h:h+step_h, w:w+step_w]
            sub_hist,sub_bins = np.histogram(sub_wordmap, bins=list(range(dict_size + 1)))   
            hist_all[i: i+dict_size] = np.divide(sub_hist, 2)    
            i+=dict_size 
    
    # compute the finer, L=1
    sub_hist =  (hist_all[0: dict_size]
                +hist_all[dict_size: dict_size*2]
                +hist_all[dict_size*4: dict_size*5]
                +hist_all[dict_size*5: dict_size*6])
    hist_all[i:i+dict_size] =  np.divide(sub_hist, 4)
    i+=dict_size
    
    sub_hist =  (hist_all[dict_size*2: dict_size*3]
                +hist_all[dict_size*3: dict_size*4]
                +hist_all[dict_size*6: dict_size*7]
                +hist_all[dict_size*7: dict_size*8])
    hist_all[i:i+dict_size] =  np.divide(sub_hist, 4)
    i+=dict_size
    
    sub_hist = (hist_all[dict_size*8: dict_size*9]
               +hist_all[dict_size*9: dict_size*10]
               +hist_all[dict_size*12: dict_size*13]
               +hist_all[dict_size*13: dict_size*14])
    hist_all[i:i+dict_size] =  np.divide(sub_hist, 4)
    i+=dict_size
    
    sub_hist = (hist_all[dict_size*10: dict_size*11]
               +hist_all[dict_size*11: dict_size*12]
               +hist_all[dict_size*14: dict_size*15]
               +hist_all[dict_size*15: dict_size*16])
    hist_all[i:i+dict_size] =  np.divide(sub_hist, 4)
    i+=dict_size
    
    # compute the coarst, L=0
    sub_hist = (hist_all[dict_size*16: dict_size*17]
               +hist_all[dict_size*17: dict_size*18]
               +hist_all[dict_size*18: dict_size*19]
               +hist_all[dict_size*19: dict_size*20])
    hist_all[i:i+dict_size] =  np.divide(sub_hist, 4)

    # L1 norm
    hist_all = np.divide(hist_all, np.sum(hist_all))
    
    return hist_all

if __name__ == '__main__':
    
    
    num_cores = util.get_num_CPU()
    
    path_img = "../data/park/labelme_evuftvyfpanmmab.jpg"
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255
    image = transform.downscale_local_mean(image, (3,3,1))
    filter_responses = visual_words.extract_filter_responses(image)
    util.display_filter_responses(filter_responses)
    compute_dictionary(num_workers=num_cores)    
    dictionary = np.load('dictionary.npy')    
    
    wordmap = get_visual_words(image,dictionary)
    filename="wordmap.jpg"
    util.save_wordmap(wordmap, filename)  
    
    build_recognition_system(num_workers=num_cores)

    conf, accuracy = evaluate_recognition_system(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())