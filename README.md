# Face-Detector-and-Classifier
Application of OpenCV to detect faces in given frame, alongwith CNN for improved accuracy and proper labelling

## Two Impletentations of OpenCV for face detection is given in this repository 
* With face cascade
* With CNN or HOG for face detection alongwith face labelling

### Cascade Implementation
Requirements:-
* Python3
* OpenCV

*Run :- python3 opencv_cascade_face_detection/face_detect.py*

### CNN / HOG face detector
Requiremnts :-
* Python3
* OpenCV
* argparse
* face_recognition
* imutils

*Build_encoder :- python3 opencv_facerecog_with_classifier/encode.py --encodings [name].pickle --dataset [dir]*

*Run :- python3 opencv_facerecog_with_classifier/recog_image.py --encodings [name].pickle --image [name].jpg*


### Example of Face Detector with CNN with trained encoder on my images

![alt text](https://raw.githubusercontent.com/Mr-Parth/Face-Detector-and-Classifier/master/opencv_facerecog_with_classifier/output/out_test.jpg )



