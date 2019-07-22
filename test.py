import cv2
import os
from mtcv import histEqualize
import numpy as np
import datetime

jpg="D:/data/1_Line17_up_20190411032439_3_34km+753.9m_forward.jpg"
bmp="D:/data/1_Line17_up_20190411034429_1370_20km+644.0m_forward.bmp"

s=datetime.datetime.now()
jpg=cv2.imread(jpg)
e=datetime.datetime.now()
print((e-s).microseconds/1000)

s=datetime.datetime.now()
bmp=cv2.imread(bmp)
e=datetime.datetime.now()
print((e-s).microseconds/1000)


a=[]
a.append([])


arr=np.arange(18).reshape((2,3,3))
b=arr[:,:2]
print(b)




# path="D:\data/visualization_leakageDataset/non-stitch"
# files=os.listdir(path)
# cv2.namedWindow("image",cv2.WINDOW_NORMAL)
# for i in files:
#     if 'uncertain' in i:
#         continue
#     img=cv2.imread(os.path.join(path,i),0)
#     img_equa=histEqualize(img,mode='clahe')
#     cv2.imshow("image",img_equa)
#     cv2.waitKey()