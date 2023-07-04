import numpy as np
from .prnu import *
from PIL import Image

def get_PRNU(image,list=False):
    k = [] # An empty list to save the fingerprints in later
    if list == True: # If provided a list of elements
        for img in image : # Iterating over all images of the current device
            if type(img) == str: # If a path was inputed instead of a Pil-Image
                im = Image.open(img) # Loading the image as a Pil Image
                im = image.resize((480,480))

            if len(im.mode) == 4:
                im = im.convert('RGB')

            # im = im.transpose(Image.ROTATE_180) # Rotating the image since there was a bug of it being upside down
            im_arr = np.asarray(im) # Loading the image as a numpy array

            # Error handeling 
            if im_arr.dtype != np.uint8: # If the image wasn't a valid image
                print('Error while reading image: {}'.format(img)) # just show the path of it
                continue # Skip to the next image
            if im_arr.ndim != 3: # If the image didn't have 3 channels (wasn't a valid RGB color or was a gray Image for example)
                print('Image is not RGB: {}'.format(img)) # just show the path of it
                continue # Skip to the next image
        
            imgs += [im_arr] # Saving the resized image in the list
            k = [extract_multiple_aligned(imgs)] # getting the Finger print of the device using the multiple Images

    else: # For a single image
        if type(image) == str: # If a path was inputed instead of a Pil-Image
            image = Image.open(image) # Loading the image as a Pil Image
            image = image.resize((480,480))
        
        if len(image.mode) == 4:
            image = image.convert('RGB')

        im_arr = np.asarray(image) # Loading the image as a numpy array

        # Error handeling 
        if im_arr.dtype != np.uint8: # If the image wasn't a valid image
            print('Error while reading image: {}'.format(img)) # just show the path of it
        if im_arr.ndim != 3: # If the image didn't have 3 channels (wasn't a valid RGB color or was a gray Image for example)
            print('Image is not RGB: {}'.format(img)) # just show the path of it

        k = [extract_single(im_arr)] # getting the Finger print of the device using a single Image

    return k
