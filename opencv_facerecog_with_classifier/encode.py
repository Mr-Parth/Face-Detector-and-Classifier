
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# parse the arguments
a = argparse.ArgumentParser()
a.add_argument("-i", "--dataset", required=True,)
a.add_argument("-e", "--encodings", required=True,)

args = vars(a.parse_args())

# grab the paths
print("fetching faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# encodings init
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("processing image {}/{}".format(i + 1,
                                          len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # BGR2RGB for dlib
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb,
                                            model="cnn")  # cnn for more accuracy but for computation

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over the encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# facial encodings + names to disk
print("encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
