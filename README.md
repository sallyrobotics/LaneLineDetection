# Lane Line Detection

This project tries to detect Lane lines on Road Images using Python and OpenCV. It works reasonably well for images with low brightness as well (after testing transformations to different color spaces). 

## Dataset
The processing has been carried out on the Caltech Pedestrian and Nexet datasets. 

The Caltech Pedestrian Dataset can be downloaded [here.](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)

The Nexet Dataset can be downloaded [here.](https://www.getnexar.com/challenge-2/)

## Usage
The code can be run using the following command:

`python3 main.py`

## Future Scope
1. Bird eye transformation for curved lane detection. (Reference: [link](https://medium.com/@tina_chien_tw/advanced-lane-finding-faa3084492eb))
2. Deep Learning based model.
