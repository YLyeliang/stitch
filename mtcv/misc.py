import cv2
import os
import numpy as np


def bgr2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def histEqualize(img,mode='clahe',clipLimit=20):
    """
    equalize histogram of a image.
    :param mode: if norm,perform normal hist, if clahe,perform adaptive histogram equalization.
    :return:
    """
    if mode =='norm':
        return cv2.equalizeHist(img)
    elif mode =='clahe':
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        return clahe.apply(img)


def read_txt_mklist(txt_path):
    bboxes_list=[]
    files=os.listdir(txt_path)
    files.sort()
    for txt in files:
        file=os.path.join(txt_path,txt)
        with open(file,'r')as f:
            bboxes=f.readlines()
            for i,box in enumerate(bboxes):
                box=box.rstrip().split(' ')
                if len(box)==1 and '' in box:
                    box=[]
                else:
                    box=[int(i) for i in box]
                bboxes[i]=box
        bboxes_list.append(bboxes)

    batches=[]
    for i in range(len(bboxes_list[0])):
        batch = []
        for j in range(len(bboxes_list)):
            box=bboxes_list[j][i]
            bboxes_tmp=[]
            for k in range(len(box) // 4):
                bboxes_tmp += [box[k * 4:k * 4 + 4]+[1]]
            batch.append(bboxes_tmp)
        batches.append(batch)

    return batches

def shift_bboxes_to_stitch(bboxes,offset_w):
    """
    add offset w to bboxes to get stitched bboxes.
    :param bboxes: (list(list)), lots of bboxes.
    :param offset_w: (int) offset w.
    :return:
    """
    if len(bboxes) ==0:
        return bboxes
    for i in range(len(bboxes)):
        xmin,ymin,xmax,ymax,score=bboxes[i]
        xmin_new=xmin+offset_w
        xmax_new=xmax+offset_w
        bboxes[i]=[xmin_new,ymin,xmax_new,ymax,score]
    return bboxes

def draw_bboxes(img,bboxes):
    for box in bboxes:
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color=(0,0,255),thickness=2)
    return img

def reshape_bboxes(bboxes):
    """
    Transform shape of (list(list)) bboxes style to (list) bboxes style.
    :param bboxes:
    :return:
    """
    box=[]
    for i in bboxes:
        for j in i:
          box+=[j]
    return box

# path = "/data2/yeliang/data/stitch_test/bbox"
# array=read_txt_mklist(path)
# # array=np.array(array)
# # print(array.shape)
# box=reshape_bboxes(array)

