import numpy as np
import matplotlib.pyplot as plt

import copy

from IO import IO as import_files
from present import present as show_figures

from skimage import io
from keras.preprocessing import image as kimage
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans as KMeans

filepath_actual = r".\translocationdata\data\*.bmp"
filepath_test = r".\test\*.bmp"

env = input("Choose current ENV [test, presentation]: ")
filepath = filepath_actual if (env == "presentation") else filepath_test

imagesChannelOne = []
imagesChannelTwo = []

imagesChannelOne, imagesChannelTwo = import_files(filepath = filepath, env = env)

show_figures(imagesChannelOne=copy.deepcopy(imagesChannelOne), imagesChannelTwo=copy.deepcopy(imagesChannelTwo), env = env)

model = VGG16(weights='imagenet', include_top=False, input_shape = (640,640,3))
model.summary()

cc = 0 
featurelist = []

while (cc < len(imagesChannelOne)):
    img = kimage.load_img(imagesChannelOne[cc], target_size=(640, 640))
    img_data = kimage.img_to_array(img)

    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())
    cc = cc + 1

kmeans = KMeans(n_clusters = 4, random_state = 0).fit(np.array(featurelist))

print(kmeans.labels_)
print(imagesChannelOne)