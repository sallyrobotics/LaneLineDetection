print("Process Image")

import numpy as np
import cv2

from utils.functions import *


def squareROI(height, width):
    # A margin of 10 pixels is kept to prevent the borders of the image 
    # being detected as an edge 
    return np.array ( [[
                    [width - 10, height / 2], 
                    [width - 10, height],
                    [10, height ],
                    [10, height / 2]
            ]], dtype = np.int32)


def trapeziumROI(height, width):
    # A margin of 40 pixels is kept to prevent the borders of the image 
    # being detected as an edge
    return np.array( [[
                [3*width/4, 3*height/5],
                [width/4, 3*height/5],
                [40, height],
                [width - 40, height]
            ]], dtype=np.int32 )


def process_image(image):

    # grayscale conversion before processing causes more harm than good
    # because sometimes the lane and road have same amount of luminance
    # grayscaleImage = grayscale(image)

    # YCrCb conversion
    # yImage = bgr_to_y(image)

    # hsv conversion
    # hsvImage = bgr_to_hsv(image)
    
    # hls conversion
    # hlsImage = bgr_to_hls(image)
    
    # applying LAB transform
    labImage = bgr_to_lab(image)

    # Blur to avoid edges from noise
    blurredImage = gaussian_blur(labImage, 11)

    # Detect edges using canny
    # high to low threshold factor of 3
    # it is necessary to keep a linient threshold at the lower end
    # to continue to detect faded lane markings
    edgesImage = canny(blurredImage, 40, 50)
    
    # mark out the trapezium region of interest
    # dont' be too agressive as the car may drift laterally
    # while driving, hence ample space is still left on both sides.
    height = image.shape[0]
    width = image.shape[1]
    vertices = trapeziumROI(height, width)

    # mask the canny output with trapezium region of interest
    regionInterestImage = region_of_interest(edgesImage, vertices)
    
    # parameters tuned using this method:
    # threshold 30 by modifying it and seeing where slightly curved 
    # lane markings are barely detected
    # min line length 20 by modifying and seeing where broken short
    # lane markings are barely detected
    # max line gap as 100 to allow plenty of room for the algo to 
    # connect spaced out lane markings
    lineMarkedImage = hough_lines(regionInterestImage, 1, np.pi/180, 40, 30, 200)
    
    # Test detected edges by uncommenting this
    #return cv2.cvtColor(regionInterestImage, cv2.COLOR_GRAY2RGB)

    # draw output on top of original
    weightedImage =  weighted_img(lineMarkedImage, image)

    return weightedImage