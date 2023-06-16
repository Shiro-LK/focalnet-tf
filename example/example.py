import cv2
import sys
import numpy as np
import os 
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from focalnet import load_focalnet, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, imagenet1k, imagenet22k
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

def preprocess_image(image ):
    #image = cv2.resize(image, input_shape)
    image = image/255.0
    image = (image - IMAGENET_DEFAULT_MEAN)/IMAGENET_DEFAULT_STD
    
    return np.expand_dims(image, axis=0)

def center_crop(image, output_shape):
    # Get the input shape
    h, w, c = image.shape

    # Get the output shape
    h_desired, w_desired = output_shape

    # Check if the output shape is valid
    if h_desired > h or w_desired > w  :
        raise ValueError("Output shape must be smaller than or equal to input shape and have the same number of channels.")

    # Compute the crop coordinates
    h_start = (h - h_desired) // 2
    h_end = h_start + h_desired
    w_start = (w - w_desired) // 2
    w_end = w_start + w_desired

    # Crop the image and return it
    return image[h_start:h_end, w_start:w_end, :]

image = cv2.cvtColor(cv2.imread("../tests/dog.jpg"), cv2.COLOR_BGR2RGB)
image_crop = center_crop(image, (768, 768))

output_shape = (224, 224)
image_resized = cv2.resize(image_crop, output_shape)
inputs = preprocess_image(image_crop)


model = load_focalnet(model_name='focalnet_tiny_srf', pretrained=True, return_model=False, act_head="softmax"   )


output  = model(inputs, training=True)

print(output[0, np.argmax(output)])
print(imagenet22k[np.argmax(output)])