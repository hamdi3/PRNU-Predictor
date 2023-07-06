import numpy as np
from .prnu import *
from PIL import Image

def get_PRNU(image, list=False):
    """
    Computes the PRNU fingerprint for an image or a list of images.

    Args:
        image (Union[str, PIL.Image.Image, List[Union[str, PIL.Image.Image]]]): 
            The input image or a list of input images.
        list (bool, optional): Flag indicating whether input is a list. Defaults to False.

    Returns:
        List[np.ndarray]: A list of PRNU fingerprints.

    Raises:
        TypeError: If the input is neither a string, PIL.Image.Image object, nor a list.
    """
    k = []  # An empty list to save the fingerprints in later

    if list:  # If provided a list of elements
        for img in image:  # Iterating over all images of the current device
            if isinstance(img, str):  # If a path was input instead of a PIL Image
                im = Image.open(img)  # Loading the image as a PIL Image
                im = image.resize((480, 480))

            if len(im.mode) == 4:
                im = im.convert('RGB')

            im_arr = np.asarray(im)  # Loading the image as a numpy array

            # Error handling
            if im_arr.dtype != np.uint8:  # If the image wasn't a valid image
                print('Error while reading image: {}'.format(img))  # Just show the path of it
                continue  # Skip to the next image
            if im_arr.ndim != 3:  # If the image didn't have 3 channels (wasn't a valid RGB color or was a gray Image for example)
                print('Image is not RGB: {}'.format(img))  # Just show the path of it
                continue  # Skip to the next image
        
            imgs += [im_arr]  # Saving the resized image in the list
            k = [extract_multiple_aligned(imgs)]  # Getting the fingerprint of the device using multiple images

    else:  # For a single image
        if isinstance(image, str):  # If a path was input instead of a PIL Image
            image = Image.open(image)  # Loading the image as a PIL Image
            image = image.resize((480, 480))
        
        if len(image.mode) == 4:
            image = image.convert('RGB')

        im_arr = np.asarray(image)  # Loading the image as a numpy array

        # Error handling 
        if im_arr.dtype != np.uint8:  # If the image wasn't a valid image
            print('Error while reading image: {}'.format(img))  # Just show the path of it
        if im_arr.ndim != 3:  # If the image didn't have 3 channels (wasn't a valid RGB color or was a gray Image for example)
            print('Image is not RGB: {}'.format(img))  # Just show the path of it

        k = [extract_single(im_arr)]  # Getting the fingerprint of the device using a single image

    return k