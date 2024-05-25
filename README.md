### COSC428 Computer Vision Assignment Okoko Anainga
### Signlanguage Detection

# Objective
The purpose of this project was to get exposure to computer vision. I chose to complete a sign language translator by using Deep learning to identify 5 signs from the American Sign Language Dictionary.

These signs were "I Love You", "Yes", "No", "Hello" and "Thank you"

For training the Neural Network for deep learning I created my own images that were captured through a web cam then I labeled them with the python image label tool.

I trained the NN with 40000 runs and 4 epochs and got down to a loss of 0.079. This seems quite effective for sign identification.
The caviate being that it did not work very well with a background of any type other than a plain colour.

# To Run
Dowload the github code and install:
    Protoc
    Tensorflow
    Python 3.10 not tested with other versions
    Cuda

# If you wish to train your own model you will also need imagelable installed

1. From here if you wish to train your own model run -> image_collection -> training -> then copy the command printed from training into a shell at the top directory of the project.

# If you wish to just run the image detection just follow part two
2. Run realTimeDetection.py
