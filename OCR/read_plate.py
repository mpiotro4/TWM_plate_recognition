import cv2
import numpy as np
from PIL import Image

from OCR.what_letter import what_letter

kernel = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])

def return_string(lista):

    lista = sorted(lista, key=lambda krotka: krotka[1])

    string = ''
    for char, _ in lista:
        string += char
    
    return string


def read_plate(image, dont_show=True):
    
    # Przekształć obraz na skalę szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # print(gray.mean())
    x = 1100/gray.shape[1]
    
    sh = tuple(int(i*x) for i in gray.shape)

    if (sh[0]<420):
        x = 420/gray.shape[0]
        sh = tuple(int(i*x) for i in gray.shape)

    gray = cv2.resize(gray, (sh[1], sh[0]))
    image = cv2.resize(image, (sh[1], sh[0]))

    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 97, 2)


    binary_image = cv2.dilate(binary_image, kernel, iterations=4)
    binary_image = cv2.erode(binary_image, kernel, iterations=4)
    binary_image = cv2.dilate(binary_image, kernel, iterations=5)
    binary_image = cv2.erode(binary_image, kernel, iterations=5)

    contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img = Image.fromarray(binary_image.astype(np.uint8))
    if not dont_show:
        img.show()
    # cv2.imshow('0', binary_image)
    # cv2.waitKey(0)

    letters = []
    # Przejdź przez wszystkie wykryte kontury
    for i, contour in enumerate(contours):

        x, y, w, h = cv2.boundingRect(contour)
        offset = int(((w+h)/2)*0.1)
        
        y1 = y-offset
        if y1<0:
            y1=0
        y2 = y+h+offset
        if y2>sh[0]:
            y2=sh[0]
        x1 = x-offset
        if x1<0:
            x1=0
        x2 = x+w+offset
        if x2>sh[1]:
            x2=sh[1]
        # Wycinaj fragment obrazu
        roi = binary_image[y1:y2, x1:x2]

        w = x2-x1
        h = y2-y1
        ratio = h/w

        length = cv2.arcLength(contour, True)

        if (length>=478 and length<=1392) and (ratio>=1.21 and ratio <= 2.78) and (w>=100 and w<=204):
    
            letter = what_letter(roi)
            letters.append((letter, x))

    string = return_string(letters)

    return string

def read_plates(plates, dont_show=True):
    strings = []
    for plate in plates:
        string = read_plate(plate, dont_show)
        strings.append(string)
    return strings

if __name__ == "__main__":
    for i in range(11):
        image = cv2.imread('blachy/blacha_'+str(i)+'.png')
        string = read_plate(image)
        print(string)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
