import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import struct
from moviepy.editor import VideoFileClip

from utils.process_image import *


# we run the algorithm on Caltech Pedestrian and Nexet Dataset
# set the corresponding directory of the dataset here
inDirectory = "Datasets/Caltech_Pedestrian"


# Nexet dataset
if(inDirectory == "Datasets/Nexet_1"):
    outDirectory = inDirectory + "_out"
    if not os.path.exists(outDirectory):
        os.makedirs(outDirectory)
    imageNames = os.listdir(inDirectory + "/")
	for imageName in imageNames:
	    if imageName == '.DS_Store':
	        continue
	    avgLeft = (0, 0, 0, 0)
	    avgRight = (0, 0, 0, 0)
	    image = mpimg.imread(inDirectory + "/" + imageName)
	    out = process_image(image)
	    mpimg.imsave(outDirectory + "/" + imageName, out)
	#     print("Processed " + outDirectory + "/" + imageName)
	print("Processing complete.")


# Caltech Pedestrian dataset
elif(inDirectory == "Datasets/Caltech_Pedestrian"):
	# reset global state of average values
	avgLeft = (0, 0, 0, 0)
	avgRight = (0, 0, 0, 0)

	''' run this code only once to convert seq files to videos'''
	# convert_seq_to_jpg('Datasets/Caltech_Pedestrian/set00')
	# convert_jpg_to_mp4('Datasets/Caltech_Pedestrian/set00_out', 'V001.mp4') 

	white_output = 'Datasets/Caltech_Pedestrian/V000_out.mp4'
	clip1 = VideoFileClip("Datasets/Caltech_Pedestrian/V000.mp4")

	'''NOTE: this function expects color images!!
	this runs the function process_image over each frame of the video'''
	white_clip = clip1.fl_image(process_image) 
	white_clip.write_videofile(white_output, audio=False)


''' Helper functions for seq file processing'''

# read seq file
def read_seq(path):
    
    def read_header(ifile):
        feed = ifile.read(4)
        norpix = ifile.read(24)
        version = struct.unpack('@i', ifile.read(4))
        length = struct.unpack('@i', ifile.read(4))
        assert(length != 1024)
        descr = ifile.read(512)
        params = [struct.unpack('@i', ifile.read(4))[0] for i in range(0,9)]
        fps = struct.unpack('@d', ifile.read(8))
        # skipping the rest
        ifile.read(432)
        image_ext = {100:'raw', 102:'jpg',201:'jpg',1:'png',2:'png'}
        return {'w':params[0],'h':params[1],
                'bdepth':params[2],
                'ext':image_ext[params[5]],
                'format':params[5],
                'size':params[4],
                'true_size':params[8],
                'num_frames':params[6]}
    
    ifile = open(path, 'rb')
    params = read_header(ifile)
    bytes = open(path, 'rb').read()

    # this is freaking magic, but it works
    extra = 8
    s = 1024
    seek = [0]*(params['num_frames']+1)
    seek[0] = 1024
    
    images = []
    
    for i in range(0, params['num_frames']-1):
        tmp = struct.unpack_from('@I', bytes[s:s+4])[0]
        s = seek[i] + tmp + extra
        if i == 0:
            val = struct.unpack_from('@B', bytes[s:s+1])[0]
            if val != 0:
                s -= 4
            else:
                extra += 8
                s += 8
        seek[i+1] = s
        nbytes = struct.unpack_from('@i', bytes[s:s+4])[0]
        I = bytes[s+4:s+nbytes]
        
        tmp_file = '/tmp/img%d.jpg' % i
        open(tmp_file, 'wb+').write(I)
        
        img = cv2.imread(tmp_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images


# convert seq to jpg images
def convert_seq_to_jpg(inDirectory):
    outDirectory = inDirectory + '_out'
    if not os.path.exists(outDirectory):
        os.makedirs(outDirectory)
    seqNames = os.listdir(inDirectory + '/')
    for seqName in seqNames:
        if seqName == '.DS_Store':
            continue
        images = read_seq(inDirectory + '/' + seqName)
        for i, image in enumerate(images):
            try:
                mpimg.imsave(outDirectory + "/" + seqName + '_' + str(i) + '.jpg', image)
            except Exception as e:
                print(e)


# to write jpg to video
def convert_jpg_to_mp4(dir_path, output):

    # Arguments
    dir_path = 'Datasets/Caltech_Pedestrian/set00_out'
    ext = 'jpg'
    output = 'Datasets/Caltech_Pedestrian/V001.mp4'

    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext) and f.startswith('V001'):
            images.append(f)

    image_dict = {}
    for image in images:
        image_dict[int(image[9:-4])] = image


    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for i in range(len(image_dict)):

    #     print(image_dict[i])
        image_path = os.path.join(dir_path, image_dict[i])
    #     print(image_path)
        frame = cv2.imread(image_path)

        out.write(frame) # Write out frame to video

        cv2.imshow('video',frame)
    #     if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
    #         break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print("The output video is {}".format(output))
  