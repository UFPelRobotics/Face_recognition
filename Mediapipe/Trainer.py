import cv2
import numpy as np
import os
import mediapipe as mp

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detect = face_detection.FaceDetection()


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        img = cv2.imread(imagePath)
        list_face = face_detect.process(img)
        if not list_face.detections:
            continue

        for face in list_face.detections:
            face_box = face.location_data.relative_bounding_box
            xmin = face_box.xmin
            ymin = face_box.ymin
            width = face_box.width
            height = face_box.height
            face_img = img[int(img.shape[0] * ymin):int(img.shape[0] * (ymin + height)),
                       int(img.shape[1] * xmin):int(img.shape[1] * (xmin + width))]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            faceSamples.append(face_img)
            ids.append(int(os.path.split(imagePath)[-1].split(".")[1]))

    return faceSamples, ids


print("\n [INFO] Treinando... Isso levará alguns segundos. Aguarde um pouco ...")

faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')
print(f"\n [INFO] {np.unique(ids)} faces classificadas. Saindo do programa")

cv2.destroyAllWindows()
