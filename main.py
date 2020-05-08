import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from keras.preprocessing import image as kimage
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans as KMeans
import glob

filepath = r".\translocationdata\data\*.bmp"
images = glob.glob(filepath)

imagesChannelOne = []
imagesChannelTwo = []


for image in images:
    if "Channel1" in image:
        imagesChannelOne.append(image)

    elif "Channel2" in image:
        imagesChannelTwo.append(image)



cc = 0
for image in imagesChannelOne:
    imagesChannelOne[cc] = io.imread(image)
    cc = cc + 1

cc = 0
for image in imagesChannelTwo:
    imagesChannelTwo[cc] = io.imread(image)
    cc = cc + 1


fullImage = []
cc = 0 
while (cc < len(imagesChannelOne)):
    full_image = np.empty((640,640,3),dtype=np.uint8)
    full_image[:,:,1] = imagesChannelOne[cc]
    full_image[:,:,2] = imagesChannelTwo[cc]
    fullImage.append(full_image)
    cc = cc + 1


plt.figure
f, axarr = plt.subplots(3,6) 
axarr[0,0].imshow(imagesChannelOne[0])
axarr[1,0].imshow(imagesChannelTwo[0])
axarr[2,0].imshow(fullImage[0])
axarr[0,1].imshow(imagesChannelOne[1])
axarr[1,1].imshow(imagesChannelTwo[1])
axarr[2,1].imshow(fullImage[1])
axarr[0,2].imshow(imagesChannelOne[2])
axarr[1,2].imshow(imagesChannelTwo[2])
axarr[2,2].imshow(fullImage[2])
axarr[0,3].imshow(imagesChannelOne[3])
axarr[1,3].imshow(imagesChannelTwo[3])
axarr[2,3].imshow(fullImage[3])
axarr[0,4].imshow(imagesChannelOne[4])
axarr[1,4].imshow(imagesChannelTwo[4])
axarr[2,4].imshow(fullImage[4])
axarr[0,5].imshow(imagesChannelOne[5])
axarr[1,5].imshow(imagesChannelTwo[5])
axarr[2,5].imshow(fullImage[5])
#plt.show()
#axarr[2].imshow(v_slice[2])

print("test")

filepath_test = r".\test\*.bmp"
images = glob.glob(filepath)

imagesChannelOne = []
imagesChannelTwo = []


for image in images:
    if "Channel1" in image:
        imagesChannelOne.append(image)

    elif "Channel2" in image:
        imagesChannelTwo.append(image)

# for image in images:
#     imagesChannelOne.append(image)

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