import cv2


def resize(self, img, ratio=0.5, dsize=None):
    """
    Resize single image to desired size.
    :param dsize: desired size.
    :param ratio: If given,image will be resize proportional to ratio.
    :return:
    """
    img_h, img_w = img.shape[0], img.shape[1]
    img_h = img_h
    img_w = img_w
    if dsize is not None:
        return cv2.resize(img, (dsize[0], dsize[1]))
    else:
        new_h, new_w = round(img_h * ratio), round(img_w * ratio)
        img = cv2.resize(img, (new_w, new_h))
        return img