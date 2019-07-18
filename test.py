import cv2
import os
from mtcv import histEqualize
import numpy as np


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