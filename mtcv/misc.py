import cv2



def bgr2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def histEqualize(img,mode='clahe'):
    """
    equalize histogram of a image.
    :param mode: if norm,perform normal hist, if clahe,perform adaptive histogram equalization.
    :return:
    """
    if mode =='norm':
        return cv2.equalizeHist(img)
    elif mode =='clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
