from imutils import face_utils
import glob
import imutils
import dlib
import cv2
import numpy as np
from sklearn.externals import joblib
import utils
from datetime import datetime
from random import shuffle

shuffle_data =True
startTime = datetime.now()
clf1 = joblib.load('../model/svm_model.pkl')  # DL Model
model = '../model/shape_predictor_68_face_landmarks.dat'

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)

#display window
win = dlib.image_window()

# load the input image, resize it, and convert it to grayscale
female_data_path = '/home/mele/datasets/gender/train/f/*.jpg'
male_data_path = '/home/mele/datasets/gender/train/m/*.jpg'

male_train_data = glob.glob(male_data_path)
female_train_data = glob.glob(female_data_path)


gender_train_data = male_train_data + female_train_data # mix the the addresses

print(len(gender_train_data))
# to shuffle data
if shuffle_data:
    shuffle(gender_train_data)

image_path = []

for i in range(len(gender_train_data)//1000 + 1): # 1000 train data
    image_path.append(gender_train_data[i])

for (k, img) in enumerate(image_path):
    image = cv2.imread(img)
    # raw_image = scipy.misc.imread(image_path)
    image = imutils.resize(image, width=200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (j, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Male/ Female Prediction
        # "Deep Learning' model
        features = utils.featureExtract(image)
        features = np.reshape(features, (1, -1))
        out = clf1.predict(features)

        if out == 0:
            text = 'Female'
        else:
            text = 'Male'
        # show the face number
        cv2.putText(image, " {}".format(text), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # print(text)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 255, 100), -1)

    # show the output image with the face detections + facial landmarks
    print('Time:', datetime.now() - startTime)
    cv2.imshow("Output", image)
    cv2.waitKey(0)