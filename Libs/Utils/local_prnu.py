import numpy as np
from Libs import prnu
from PIL import Image
import pandas as pd

def save_dict_to_excel(dictionary, filename):
    """
    Saves a dictionary into an Excel file with formatted cells.

    Args:
        dictionary (dict): The dictionary to be saved.
        filename (str): The name of the output Excel file.

    Returns:
        None

    Raises:
        IOError: If there is an error while writing the Excel file.

    Example:
        my_dict = {'Name': 'John', 'Age': 30, 'Location': 'New York'}
        save_dict_to_excel(my_dict, 'output.xlsx')
    """
    df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Value'])
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    df.to_excel(writer, sheet_name='Sheet1', index_label='Key', freeze_panes=(1, 1))
    
    # Adjusting column widths to fit the content
    worksheet = writer.sheets['Sheet1']
    for column in worksheet.columns:
        max_length = 0
        column = [cell for cell in column]
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(cell.value)
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    
    writer.save()

def save_dict_as_csv(data_dict, file_path):
    """
    Save a dictionary as a CSV file with ";" as the separator.

    Arguments:
    - data_dict: A dictionary containing the data to be saved.
    - file_path: The file path where the CSV file should be saved.
    """
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df.to_csv(file_path, sep=';')
    print(f"Dictionary saved as CSV file: {file_path}")

def get_PRNU(image,list=False):
    k = [] # An empty list to save the fingerprints in later
    if list == True: # If provided a list of elements
        for img in image : # Iterating over all images of the current device
            if type(img) == str: # If a path was inputed instead of a Pil-Image
                im = Image.open(img) # Loading the image as a Pil Image

            im = im.transpose(Image.ROTATE_180) # Rotating the image since there was a bug of it being upside down
            im_arr = np.asarray(im) # Loading the image as a numpy array

            # Error handeling 
            if im_arr.dtype != np.uint8: # If the image wasn't a valid image
                print('Error while reading image: {}'.format(img)) # just show the path of it
                continue # Skip to the next image
            if im_arr.ndim != 3: # If the image didn't have 3 channels (wasn't a valid RGB color or was a gray Image for example)
                print('Image is not RGB: {}'.format(img)) # just show the path of it
                continue # Skip to the next image
        
            im_cut = prnu.cut_ctr(im_arr, (480, 480, 3)) # Resizing the image as was recommended
            imgs += [im_cut] # Saving the resized image in the list
            k = [prnu.extract_multiple_aligned(imgs)] # getting the Finger print of the device using the multiple Images

    else: # For a single image
        if type(image) == str: # If a path was inputed instead of a Pil-Image
            image = Image.open(image) # Loading the image as a Pil Image

        image = image.transpose(Image.ROTATE_180) # Rotating the image since there was a bug of it being upside down
        im_arr = np.asarray(image) # Loading the image as a numpy array

        # Error handeling 
        if im_arr.dtype != np.uint8: # If the image wasn't a valid image
            print('Error while reading image: {}'.format(image)) # just show the path of it
        if im_arr.ndim != 3: # If the image didn't have 3 channels (wasn't a valid RGB color or was a gray Image for example)
            print('Image is not RGB: {}'.format(image)) # just show the path of it

        im_cut = prnu.cut_ctr(im_arr, (480, 480, 3)) # Resizing the image as was recommended
        k = [prnu.extract_single(im_cut)] # getting the Finger print of the device using a single Image

    return k
