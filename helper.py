import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

def plot_image(img, title ='data', save=False, save_path = None):
    """
    Plots the steering data set distribution
    """
    #print(type(img))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    
    if(save):
        fig = plt.gcf()
        
    plt.show()
    plt.draw()
    plt.pause(0.05)
    if(save):
        if(save_path is None): raise Exception('"save_pathe" is None!')
        fig.savefig(save_path, dpi=300)
        