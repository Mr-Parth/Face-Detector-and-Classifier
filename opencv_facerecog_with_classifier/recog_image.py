# USAGE
# python recog_image.py --encodings test.pickle --image test2.jpg


import face_recognition
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                )
ap.add_argument("-i", "--image", required=True,
                )
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="either `hog` or `cnn`")  # hog :- less acc and less computation ; cnn :- more computation more accuracy
args = vars(ap.parse_args())

# load the known faces and embeddings
print("loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# recog
print("recognizing faces...")
boxes = face_recognition.face_locations(rgb,
                                        model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []

# loop over
for encoding in encodings:

    matches = face_recognition.compare_faces(data["encodings"],
                                             encoding)
    name = "Unavailable"

    # check to see if we have found a match
    if True in matches:

        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        name = max(counts, key=counts.get)

    # update the list of names
    names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
    # draw the predicted face name on the image
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

# show the output image
cv2.imshow("Image", image)
cv2.imwrite("output/out_test.jpg", image)
cv2.waitKey(0)
