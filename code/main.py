import numpy as np
import util
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import visual_words
import visual_recog
import skimage.io


if __name__ == '__main__':
    
    
    num_cores = util.get_num_CPU()
    
    path_img = "../data/ISIC_2019_Training_Input/ISIC_0000000.jpg"
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255
    filter_responses = visual_words.extract_filter_responses(image)
    util.display_filter_responses(filter_responses)
    

    visual_words.compute_dictionary(num_workers=num_cores)
    
    dictionary = np.load('dictionary.npy')
    
    wordmap = visual_words.get_visual_words(image, dictionary)
    #print(wordmap)
    #np.save("wordmap.npy",wordmap)  
    #wordmap = np.load("wordmap.npy")
    
    filename="wordmap.jpg"
    util.save_wordmap(wordmap, filename)
    #a = visual_recog.get_feature_from_wordmap(wordmap, 100)
    #b = visual_recog.get_feature_from_wordmap_SPM(wordmap,3, 100)
    
    
    #######
    visual_recog.build_recognition_system(num_workers=num_cores)

    conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())
    
    

