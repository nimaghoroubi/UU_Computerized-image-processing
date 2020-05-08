from skimage import io
import numpy as np
import matplotlib.pyplot as plt

def present(imagesChannelOne, imagesChannelTwo, env):
    
    if env == "presentation":
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

        plt.show()
     
    else:
        print("Presentation not set for Test environment")