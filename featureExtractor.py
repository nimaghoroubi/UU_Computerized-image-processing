from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input


def GetFeatures(imagesChannelOne, model):
    cc = 0 
    featurelist = []

    while (cc < len(imagesChannelOne)):
        img = image.load_img(imagesChannelOne[cc], target_size=(640, 640))
        img_data = image.img_to_array(img)

        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = np.array(model.predict(img_data))
        featurelist.append(features.flatten())
        cc = cc + 1

    return featurelist