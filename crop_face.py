from mtcnn import MTCNN
import cv2
import numpy as np
from PIL import Image

def crop_face(filename, required_size=(224, 224)):
    img = cv2.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']
    face = img[y:y+height, x:x+width]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array, face

if __name__ == "__main__":
    a, b = crop_face("./test/angry/im14.png")
    cv2.imshow("Display1",a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()