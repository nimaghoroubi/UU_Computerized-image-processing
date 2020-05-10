import copy

from IO import IO as import_files
from present import present as show_figures
from featureExtractor import GetFeatures

from skimage import io
from sklearn.cluster import KMeans as KMeans
from keras.applications.vgg16 import VGG16
import numpy as np


filepath_actual = r".\translocationdata\data\*.bmp"
filepath_test = r".\test\*.bmp"

env = input("Choose current ENV [test, presentation]: ")
filepath = filepath_actual if (env == "presentation") else filepath_test

imagesChannelOne = []
imagesChannelTwo = []

# import files based on environment we are in
imagesChannelOne, imagesChannelTwo = import_files(filepath = filepath, env = env)

# show the figures if we are in presentation
show_figures(imagesChannelOne=copy.deepcopy(imagesChannelOne), imagesChannelTwo=copy.deepcopy(imagesChannelTwo), env = env)

# create a model for feature extraction
model = VGG16(weights='imagenet', include_top=False, input_shape = (640,640,3))
model.summary()

# extract features based on model and image input, feature list is a list of feature datastructures for each image
featurelist = GetFeatures(imagesChannelOne, model)

# use kmeans to categorize the features
kmeans = KMeans(n_clusters = 4, random_state = 0).fit(np.array(featurelist))

# save 
print(kmeans.labels_)
print(imagesChannelOne)