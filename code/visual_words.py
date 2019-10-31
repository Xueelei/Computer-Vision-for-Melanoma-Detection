import numpy as np
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import random


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
    filter_response = extract_filter_responses(image)
    
    ''' get random pixels '''
    #alpha_response = (np.random.permutation(filter_response))
    alpha_response = np.zeros(shape = (alpha,filter_response.shape[2]))
    for j in range(alpha):
        row = np.random.randint(0,filter_response.shape[0])
        col = np.random.randint(0,filter_response.shape[1])
        alpha_response[j]=filter_response[row][col]

    np.save(str(i)+".npy",alpha_response)

