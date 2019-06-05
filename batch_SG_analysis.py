
from nd2reader import ND2Reader
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import numpy as np
import seaborn as sn
import sys
from tifffile import TiffFile

class SgFinder:
    def __init__(self, im_path, rG_background, haloTag_channel=2, rG_channel=1):
        if im_path[-4:] == '.nd2':
            self.image = []
            with ND2Reader(im_path) as nd2:
                for channel in range(3):
                    self.image.append(nd2[channel])
        elif im_path[-4:] == '.tif':
            with TiffFile(im_path) as tif:
                self.image = tif.asarray()
        else:
            print("check yo extension. I found '{}'. I can only use '.nd2' or '.tif'.".format(im_path[-4:]))
        self.haloTag_channel = haloTag_channel
        self.rG_channel = rG_channel
        self.rG_background = rG_background
        self.haloTag_threshold = None
        self.granule_count = None
        self.granule_mask = None
        self.ratios = []
    
    def median_filter(self, pixels):
        for channel in range(3):
            self.image[channel] = ndi.median_filter(self.image[channel], size=pixels)
    
    def setHaloThreshold(self, threshold=None, slider_range=(0,3000)):
        
        def t(threshold_value):
            t_img = self.image[self.haloTag_channel] > threshold_value
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(t_img)
            #plt.imshow(t_img)
            plt.show()
            labeled_sg, count = ndi.label(self.image[self.haloTag_channel]>threshold_value)
            print("found {}".format(count))
            self.haloTag_threshold = threshold_value
            self.granule_count = count
            self.granule_mask = labeled_sg
            
        if threshold is not None:
            self.haloTag_threshold = threshold
            slider_value = threshold
            t(threshold)
        else:
            slider_value = 0
#         def t(threshold_value):
#             t_img = self.image[self.haloTag_channel] > threshold_value
#             fig, ax = plt.subplots(figsize=(10, 10))
#             ax.imshow(t_img)
#             #plt.imshow(t_img)
#             plt.show()
#             labeled_sg, count = ndi.label(self.image[self.haloTag_channel]>threshold_value)
#             print("found {}".format(count))
#             self.haloTag_threshold = threshold_value
#             self.granule_count = count
#             self.granule_mask = labeled_sg

            slider = widgets.IntSlider(min=slider_range[0], max=slider_range[1], step=10, value=slider_value)
            interact(t, threshold_value=slider)
        
    def dilate_and_ratio(self, iterations=3, sig_noise_requirement=3):
        self.ratios = []
        for label in range(1,self.granule_count):
            print("completed {} of {}".format(label, self.granule_count), end="\r")
            mask = self.granule_mask==label
            rg_sg_mask = np.ma.array(self.image[self.rG_channel], mask=np.invert(mask))
            sg_median = np.ma.median(rg_sg_mask)
            if sg_median<sig_noise_requirement*self.rG_background: continue
            dilation = ndi.morphology.binary_dilation(mask, iterations=iterations) # might have to adjust iterations
            outline = np.logical_xor(dilation, mask)
#            rg_sg_mask = np.ma.array(self.image[self.rG_channel], mask=np.invert(mask))
            rg_outline_mask = np.ma.array(self.image[self.rG_channel], mask=np.invert(outline))
#            sg_median = np.ma.median(rg_sg_mask)
            outline_median = np.ma.median(rg_outline_mask)
            if (sg_median<sig_noise_requirement*self.rG_background) or (outline_median<sig_noise_requirement*self.rG_background): continue
            self.ratios.append(sg_median/outline_median)
        
        print("found {} loaded particles. \n Average Ratio: {}".format(len(self.ratios), np.mean(self.ratios)))