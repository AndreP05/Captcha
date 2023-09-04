import os
import shutil
import math
import cv2
import numpy as np
import pytesseract
from glob import glob
from PIL import Image

kernel = np.array([[-1, -1, -1],
                   [-1, 1, -1],
                   [-1, -1, -1]], dtype="int")


def threshold_plus(image):
    # Inverse threshold
    _, thresh = cv2.threshold(image, 163, 255, cv2.THRESH_BINARY_INV)

    # Remove colors except black and white
    for y in thresh:
        for x in y:
            if not (x[0] == 255 and x[0] == x[1] and x[0] == x[2]):
                if not (x[0] == 0 and x[0] == x[1] and x[0] == x[2]):
                    x[0] = 0
                    x[1] = 0
                    x[2] = 0

    # Remove pixels away from components
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    single_pixels = cv2.morphologyEx(thresh, cv2.MORPH_HITMISS, kernel)
    single_pixels_inv = cv2.bitwise_not(single_pixels)

    return cv2.bitwise_and(thresh, thresh, mask=single_pixels_inv)


def relevant_connections(image):
    connectivity = 8  # 4 ou 8
    delete = []

    # Convert to gray scale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    num_labels, cc_image, stats, _ = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    # Remove irrelevant components from analysis
    for x in range(num_labels):
        if stats[x][4] < 10:
            delete.append(x)
    for x in delete:
        num_labels -= 1
        np.delete(cc_image, x)
        np.delete(stats, x)

    return num_labels, cc_image, stats


def border_image(image, size):
    # Configure test images
    extra = size - image.shape[1]

    if not len(image.shape) < 3:
        digit = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        digit = image

    if extra % 2 == 0:
        digit = np.hstack((np.zeros((digit.shape[0], int(extra / 2))), digit))
        digit = np.hstack((digit, np.zeros((digit.shape[0], int(extra / 2)))))
    else:
        digit = np.hstack((np.zeros((digit.shape[0], int(math.floor(extra / 2)))), digit))
        digit = np.hstack((digit, np.zeros((digit.shape[0], int(math.ceil(extra / 2))))))

    digit = np.vstack((np.zeros((1, digit.shape[1])), digit))
    digit = np.vstack((digit, np.zeros((1, digit.shape[1]))))

    digit = cv2.resize(digit, (size, size))

    return digit


def break_captcha(imagem, mode=0):
    # Reading
    cap = cv2.imread(imagem)
    result = []

    # Thresholding and Removing Colors
    thresh = threshold_plus(cap)

    # Getting relevant connected elements
    num_labels, cc_image, stats = relevant_connections(thresh)

    if mode == 0:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # The first component is always the whole image
        x, y, w, h = stats[0][:4]
        component_threshold = thresh[y:y + h, x:x + w]
        cv2.imwrite(os.getcwd() + "\\images\\complete.png", component_threshold)

        # Try full image on tesseract
        result.append(pytesseract.image_to_string(
            Image.open('images\\complete.png'),
            lang='eng',
            config='--oem 3 -c tessedit_char_whitelist=0123456789').strip())

    elif mode == 1:
        # Try resized images on knn model
        knn = cv2.ml.KNearest_load('models\\knn_model.xml')
        test_cells = []

        for i in range(1, num_labels):
            x, y, w, h = stats[i][:4]

            component_threshold = thresh[y:y + h, x:x + w]
            img = border_image(component_threshold, 21)

            cv2.imwrite(os.getcwd() + "\\images\\resized\\sample_" + str(i - 1) + ".png", img)

            test_cells.append(img.flatten())
        test_cells = np.array(test_cells, dtype=np.float32)

        _, result, _, _ = knn.findNearest(test_cells, k=1)

    return result


def generate_train_base():
    c = []
    c_labels = []

    # Create array of flatten images and its labels
    for file in list(glob("images\\train_base\\*.png")):
        if "zero" in file:
            c_labels.append(0)
        elif "one" in file:
            c_labels.append(1)
        elif "two" in file:
            c_labels.append(2)
        elif "three" in file:
            c_labels.append(3)
        elif "four" in file:
            c_labels.append(4)
        elif "five" in file:
            c_labels.append(5)
        elif "six" in file:
            c_labels.append(6)
        elif "seven" in file:
            c_labels.append(7)
        elif "eight" in file:
            c_labels.append(8)
        elif "nine" in file:
            c_labels.append(9)

        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (21, 21))
        c.append(img.flatten())
    c = np.array(c, dtype=np.float32)
    c_labels = np.array(c_labels, dtype=np.float32)

    return c, c_labels


def learn_captcha():
    cells, cells_labels = generate_train_base()

    # KNN
    knn = cv2.ml.KNearest_create()
    knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
    knn.save('models\\knn_model.xml')


def register_success(result):
    # Upgrade train base when successful break
    for i in range(len(result)):
        if result[i] == "0":
            last = len(glob('images\\train_base\\zero*'))
            shutil.copyfile("images\\resized\\sample_{}.png".format(i), "images\\train_base\\zero_{}.png".format(last))
        elif result[i] == "1":
            last = len(glob('images\\train_base\\one*'))
            shutil.copyfile("images\\resized\\sample_{}.png".format(i), "images\\train_base\\one_{}.png".format(last))
        elif result[i] == "2":
            last = len(glob('images\\train_base\\two*'))
            shutil.copyfile("images\\resized\\sample_{}.png".format(i), "images\\train_base\\two_{}.png".format(last))
        elif result[i] == "3":
            last = len(glob('images\\train_base\\three*'))
            shutil.copyfile("images\\resized\\sample_{}.png".format(i), "images\\train_base\\three_{}.png".format(last))
        elif result[i] == "4":
            last = len(glob('images\\train_base\\four*'))
            shutil.copyfile("images\\resized\\sample_{}.png".format(i), "images\\train_base\\four_{}.png".format(last))
        elif result[i] == "5":
            last = len(glob('images\\train_base\\five*'))
            shutil.copyfile("images\\resized\\sample_{}.png".format(i), "images\\train_base\\five_{}.png".format(last))
        elif result[i] == "6":
            last = len(glob('images\\train_base\\six*'))
            shutil.copyfile("images\\resized\\sample_{}.png".format(i), "images\\train_base\\six_{}.png".format(last))
        elif result[i] == "7":
            last = len(glob('images\\train_base\\seven*'))
            shutil.copyfile("images\\resized\\sample_{}.png".format(i), "images\\train_base\\seven_{}.png".format(last))
        elif result[i] == "8":
            last = len(glob('images\\train_base\\eight*'))
            shutil.copyfile("images\\resized\\sample_{}.png".format(i), "images\\train_base\\eight_{}.png".format(last))
        elif result[i] == "9":
            last = len(glob('images\\train_base\\nine*'))
            shutil.copyfile("images\\resized\\sample_{}.png".format(i), "images\\train_base\\nine_{}.png".format(last))
