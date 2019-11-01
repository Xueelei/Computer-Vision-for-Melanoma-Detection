import numpy as np
import threading
import queue
import imageio
import os,time
import visual_words
import skimage.io
import multiprocessing
import datetime


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
    SPM_layer_num = 3
    
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
        print(i)
    
    
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
        print("accurate:",test_labels[i],"predict:", label, "accuracy:",accuracy)    
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
    image = skimage.io.imread('../data/ISIC_2019_Training_Input/'+file_path)
    image = image.astype('float')/255
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






    

