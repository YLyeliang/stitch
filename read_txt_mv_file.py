import os
import shutil
import cv2

# 读取txt文件
# file = 'D:/data/stitch/crop/ImageSets/Main/test.txt'
# img_path = "/data/crop/VOC2007/JPEGImages"
# files=[]
# imgs=[]
# with open(file, 'r') as f:
#     while True:
#         line = f.readline()
#         line = line[:-1]
#         files.append(line+'.jpg')
#         if not line:
#             break
#         src = os.path.join(img_path, line) + ".jpg"
#         imgs.append(src)

# path = '/data2/yeliang/data/leakage_test'
# # files= os.listdir(path)
# imgs = [os.path.join(path,i) for i in files]

debug=1

def read_txt_cp_filev1(txt,src_path,dst_path):
    """ used to copy files in voc format txt file.
    which image name have no postfix like jpg"""
    with open(txt, 'r') as f:
        while True:
            line = f.readline()
            line = line[:-1]
            if not line:
                break
            src = os.path.join(src_path, line) + ".jpg"
            dst = os.path.join(dst_path, line) + ".jpg"
            shutil.copy(src, dst)




def read_txt_cp_filev2(txt,src_path,dst_path):
    """ used to copy files in normal txt file.
    which have name like Camera{}/xxx.jpg/bmp"""
    with open(txt, 'r') as f:
        while True:
            line = f.readline().rstrip()
            if not line:
                break
            line=line.split(' ')[0].split('/')[1]
            line = line[:-3]+'bmp'
            src = os.path.join(src_path, line)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            dst = os.path.join(dst_path, line)
            shutil.copy(src, dst)

def bmp2jpg():
    for i in range(1,8):
        yes_path = "/data2/yeliang/data/tunnel_camera/20190719_line10_camera/20190719_model/source/Camera{}_yes".format(
            i)
        no_path = "/data2/yeliang/data/tunnel_camera/20190719_line10_camera/20190719_model/source/Camera{}_no".format(i)
        no_imgs=os.listdir(no_path)
        no_imgs.sort()
        yes_imgs_ = os.listdir(yes_path)
        yes_imgs=[]
        for j in yes_imgs_:
            if not 'xml' in j:
                yes_imgs.append(j)
        yes_imgs.sort()
        for yes in yes_imgs:
            img=cv2.imread(os.path.join(yes_path,yes))
            jpg_out=os.path.join(yes_path,yes[:-3]+'jpg')
            bmp_remove=os.path.join(yes_path,yes)
            cv2.imwrite(jpg_out,img)
            os.remove(bmp_remove)

        for no in no_imgs:
            img=cv2.imread(os.path.join(no_path,no))
            jpg_out=os.path.join(no_path,no[:-3]+'jpg')
            bmp_remove=os.path.join(no_path,no)
            cv2.imwrite(jpg_out,img)
            os.remove(bmp_remove)


def cp_file():
    for i in range(1,8):
        img_path="/data2/yeliang/data/leakage_data/line_10_719/Camera{}".format(i)
        yes_path="/data2/yeliang/data/tunnel_camera/20190719_line10_camera/20190719_model/source/Camera{}_yes".format(i)
        no_path = "/data2/yeliang/data/tunnel_camera/20190719_line10_camera/20190719_model/source/Camera{}_no".format(i)
        yes_txt='/data2/yeliang/data/tunnel_camera/20190719_line10_camera/20190719_model/Camera{}_yes.txt'.format(i)
        no_txt='/data2/yeliang/data/tunnel_camera/20190719_line10_camera/20190719_model/Camera{}_no.txt'.format(i)
        read_txt_cp_filev2(yes_txt,img_path,yes_path)
        read_txt_cp_filev2(no_txt,img_path,no_path)

bmp2jpg()

