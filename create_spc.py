#### Code to create synthetic Binary Images from RGB Images ####
### Just input the folder in BASE and it will output the binary images in BASE__1 ###
import numpy as np
import os, sys
import cv2
# Original Clean Images
BASE = '/home/vinayak/DLP-Kaggle/EE5178/data/testing-images'
frames = [1]

TOTAL_FRAMES=frames[-1]
imgs = sorted(os.listdir(BASE))
for idx in range(len(imgs)):
    im = cv2.imread(os.path.join(BASE, imgs[idx]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    count_shape = im.shape + (-1,)
    
    for i in range(1):
        # Simulate low light conditions by scaling pixel values by 1/1000.
        # Poisson process to simulate photon arrival
        photon_counts = np.random.poisson(im.flatten()/255., size=(TOTAL_FRAMES, im.size)).T
        photon_counts = photon_counts.reshape(count_shape)
        # Photon counts is converted to binary frames
        
        b_counts=np.where(photon_counts>=1, 1, 0)
        for fil in frames:
            recon_image = np.mean(b_counts[:,:,0:fil], axis=2)
            #outfile = BASENEW.replace('60', str(fil)) + str(i) + im_name.replace('JPEG', 'png')
            outfile = os.path.join(BASE+"__"+str(fil), imgs[idx])
            directory = os.path.dirname(outfile)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # recon_image = np.repeat(np.expand_dims(recon_image, axis = -1), 3, axis = -1)
            recon_image = np.expand_dims(recon_image, axis = -1)
            cv2.imwrite(outfile, recon_image*255.)