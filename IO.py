import glob
import sys

def IO(filepath, env):
    images = glob.glob(filepath)

    imagesChannelOne = []
    imagesChannelTwo = []

    if env == "test":
        for image in images: 
            imagesChannelOne.append(image)

    elif env == "presentation":
        for image in images:
            if "Channel1" in image:
                imagesChannelOne.append(image)

            elif "Channel2" in image:
                imagesChannelTwo.append(image)
    else:
        print ("Wrong Input Environment, the environment does not exist.")
        sys.exit(1)
    return imagesChannelOne, imagesChannelTwo