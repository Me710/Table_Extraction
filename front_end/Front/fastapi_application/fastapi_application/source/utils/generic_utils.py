import io
import numpy as np
from PIL import Image
from fastapi import UploadFile, File
import cv2 as cv
import pandas as pd
#import pytesseract
import pandas as pd
import torch
from torchvision.ops import nms
import numpy as np

def image_input(image_content: bytes):
    image_from_memory = np.asarray(bytearray(io.BytesIO(image_content).read()), dtype="uint8")
    decoded_image_bgr = cv.imdecode(image_from_memory, cv.IMREAD_COLOR)
    decoded_image_rgb = cv.cvtColor(decoded_image_bgr, cv.COLOR_BGR2RGB)
    return decoded_image_rgb


def get_bytestream_from_image(image: np.ndarray) -> io.BytesIO:
    """
    Return an image bytestream from an image array
    :param image: image array
    :return: byte stream of image
    """
    preprocessed_image = Image.fromarray(image)
    return_image = io.BytesIO()
    preprocessed_image.save(return_image, "PNG")
    return_image.seek(0)
    return return_image

def extract_information_from_boxes(image, plotted_boxes):
    data = []
    for box in plotted_boxes:
        xmin, ymin, xmax, ymax = box
        cropped_img = image.crop((xmin, ymin, xmax, ymax))  # Crop the image based on the box coordinates
        ocr_result = pytesseract.image_to_string(cropped_img)  # Perform OCR on the cropped image
        data.append({'Bounding Box': box, 'OCR Result': ocr_result.strip()})

    df = pd.DataFrame(data)
    return df


def save_dataframe_to_file(df, file_path, file_format='csv'):
    """
    Save a DataFrame to an Excel or CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to the output file.
        file_format (str, optional): The file format to use ('excel' or 'csv'). Default is 'csv'.
    """
    if file_format.lower() == 'excel':
        df.to_excel(file_path, index=False)
        print(f'DataFrame saved to Excel file: {file_path}')
    elif file_format.lower() == 'csv':
        df.to_csv(file_path, index=False)
        print(f'DataFrame saved to CSV file: {file_path}')
    else:
        print(f'Unsupported file format: {file_format}. Please use "excel" or "csv".')


def crop_object(image, box):
  """Crops an object in an image

  Inputs:
    image: PIL image
    box: one box from Detectron2 pred_boxes
  """
  crop_img = image
  print("Trueeeeee")
  x_top_left = box[0][0]
  y_top_left = box[0][1]
  x_bottom_right = box[0][2]
  y_bottom_right = box[0][3]
  x_center = (x_top_left + x_bottom_right) / 2
  y_center = (y_top_left + y_bottom_right) / 2

  #crop_img = image.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))
  crop_img = image[int(y_top_left):int(y_bottom_right) + 1, int(x_top_left):int(x_bottom_right) + 1]
  return crop_img