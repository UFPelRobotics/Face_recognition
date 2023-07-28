import cv2
import numpy as np
import os
import mediapipe as mp

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detect = face_detection.FaceDetection(min_detection_confidence=0.5)


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        print(imagePath)
        img = cv2.imread(imagePath)
        if img is None:
            print(f"[AVISO] ERRO AO LER IMAGEM: {imagePath}")
            continue

        results = face_detect.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.detections:
            print(f"[AVISO] NENHUM ROSTO DETECTADO: {imagePath}")
            continue

        for face in results.detections:
            face_box = face.location_data.relative_bounding_box
            xmin, ymin, width, height = face_box.xmin, face_box.ymin, face_box.width, face_box.height
            face_img = img[int(img.shape[0] * ymin):int(img.shape[0] * (ymin + height)),
                       int(img.shape[1] * xmin):int(img.shape[1] * (xmin + width))]

            # Ensure that face_img is not empty before converting to grayscale
            if face_img.size == 0:
                print(f"[AVISO] Sem rosto em:{imagePath}!")
            else:
                print("else")
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) 
                faceSamples.append(face_img)
                #ids.append(int(os.path.split(imagePath)[-1].split(".")[1]))
                # print(os.path.split(imagePath)[-1].split("_")[1])

                ids.append(0)

    return faceSamples, ids


print("\n [INFO] Training")

faces, ids = getImagesAndLabels(path)
print(len(faces),len(ids))
if len(faces) == 0:
    print("[AVISO] NÃO FORAM ENCONTRADOS ROSTOS PARA REALIZAR O TREINO.")
else:
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')
    print(f"\n [INFO] {len(np.unique(ids))} CLASSIFICADO. FECHANDO O PROGRAMA")

cv2.destroyAllWindows()


