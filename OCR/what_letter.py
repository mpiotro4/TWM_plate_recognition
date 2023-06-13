from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

img_height = 128
img_width = 128

reverse_mapping = {0: '0',
                   1: '1',
                   2: '2',
                   3: '3',
                   4: '4',
                   5: '5',
                   6: '6',
                   7: '7',
                   8: '8',
                   9: '9',
                   10: 'A',
                   11: 'B',
                   12: 'C',
                   13: 'D',
                   14: 'E',
                   15: 'F',
                   16: 'G',
                   17: 'H',
                   18: 'I',
                   19: 'J',
                   20: 'K',
                   21: 'L',
                   22: 'M',
                   23: 'N',
                   24: 'O',
                   25: 'P',
                   26: 'Q',
                   27: 'R',
                   28: 'S',
                   29: 'T',
                   30: 'U',
                   31: 'V',
                   32: 'W',
                   33: 'X',
                   34: 'Y',
                   35: 'Z'}


model = load_model('OCR/0_model_0083.h5')

def what_letter(img):
    # Przekształcenie obrazu do tablicy numpy i dodanie dodatkowego wymiaru
    img = img_to_array(img)
    img = cv2.resize(img, (img_height, img_width))
    img = img.reshape((1, img_height, img_width, 1))

    # Skalowanie pikseli do zakresu 0-1
    img = img / 255.

    # Predykcja klasy obrazu
    prediction = model.predict(img)

    # Wybór klasy z największym prawdopodobieństwem
    predicted_class_index = prediction.argmax(axis=-1)

    # Przekształcenie indeksu na etykietę klasy
    predicted_class_label = reverse_mapping[predicted_class_index[0]]

    return(predicted_class_label)