
import cv2
from  faceTraining import recognizer

recognizer.read('trainer/trainer.txt')

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


font = cv2.FONT_HERSHEY_SIMPLEX


names = ['none', 'Meric', 'none', 'Rose', '']


cam = cv2.VideoCapture(0)
cam.set(3, 1000)
cam.set(4, 800)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(

        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])



        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "Unknown"
            confidence = "  {0}%".format(round(100 - confidence))



        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)


        cv2.putText(img, str(confidence), (x - 20, y + h - 5), font, 1, (255, 255, 0), 1)
                                            # +5
    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff
    if k == 27 or k == ord('q'):
        break

print("\n [INFO] I exit the program and clear the memory")
cam.release()
cv2.destroyAllWindows()






