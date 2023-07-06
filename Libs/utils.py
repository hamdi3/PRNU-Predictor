import torch
import numpy as np
from .local_prnu import get_PRNU
from .model import model

label_dict = {'FrontCamera-GalaxyA13-225951': 0, 'FrontCamera-GalaxyA13-225952': 1, 'Logitech Brio210500': 2,
              'Logitech Brio210504': 3, 'Logitech Brio210506': 4, 'Logitech C50596011268': 5, 'Logitech C50596011268_2': 6,
              'Logitech C50596011268_3': 7, 'Nikon_Zfc': 8, 'RückCamera-GalaxyA13-225951': 9, 'Rückcamera-GalaxyA13-225952': 10}

def get_key_from_value(dictionary, value):
    """
    Retrieves the key corresponding to a given value in a dictionary.

    Args:
        dictionary (Dict[Any, Any]): The input dictionary.
        value (Any): The value for which the key is to be retrieved.

    Returns:
        Any: The key corresponding to the given value, or None if not found.
    """
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def predict_image(img, model=model, label_dict=label_dict):
    """
    Predicts the label of an image using the provided model and label dictionary.

    Args:
        img (PIL.Image.Image): The input image.
        model (torch.nn.Module, optional): The trained model to use for prediction. Defaults to the pre-defined model.
        label_dict (Dict[str, int], optional): The dictionary mapping labels to integers. Defaults to the pre-defined label dictionary.

    Returns:
        str: The predicted label of the image.
    """
    img = img.resize((480, 480))  # Resizing the image before prediction
    image_prnu = torch.tensor(np.array(get_PRNU(img)))
    prediction = model(image_prnu)
    _, predicted = torch.max(prediction, 1)
    return get_key_from_value(label_dict, predicted.item())