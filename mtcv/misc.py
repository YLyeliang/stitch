import cv2
import os



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
        clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
        return clahe.apply(img)


def read_txt_mklist(txt_path):
    bboxes_list=[]
    files=os.listdir(txt_path)
    for txt in files:
        file=os.path.join(txt_path,txt)
        with open(file,'r')as f:
            bboxes=f.readline().rstrip().split()
            bboxes=[int(i) for i in bboxes]
            bboxes_tmp=[]
            for i in range(len(bboxes)//4):
                bboxes_tmp+=[bboxes[i*4:i*4+4]]
        bboxes_list.append(bboxes_tmp)
    return bboxes_list

