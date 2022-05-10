import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'
# eigner &  fisher alg
recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");



def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    faceSamples=[]
    ids = []



    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') .


        img_numpy = np.array(PIL_img)           # ,'uint8'



        id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces = detector.detectMultiScale(img_numpy)

        for x in faces:
            faceSamples.append(img_numpy)
            ids.append(id)

    return faceSamples,ids


print ("\n [INFO] Faces Training, Wait a few seconds ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.save('trainer/trainer.txt')

#

print("\n [INFO] {0} faces training. Exit ".format(len(np.unique(ids))))



