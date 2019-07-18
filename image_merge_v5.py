#!/usr/bin/env python3
import cv2
import numpy as np
import random
import math
import os
import datetime
from scipy.interpolate import RectBivariateSpline
from mtcv import histEqualize
# opencv-contrib-python 3.4.2.16
# opencv-python         3.4.5.20

# 拼接连接处处理
# //优化两图的连接处，使得拼接自然
# void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
# {
#     int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界
#
#     double processWidth = img1.cols - start;//重叠区域的宽度
#     int rows = dst.rows;
#     int cols = img1.cols; //注意，是列数*通道数
#     double alpha = 1;//img1中像素的权重
#     for (int i = 0; i < rows; i++)
#     {
#         uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
#         uchar* t = trans.ptr<uchar>(i);
#         uchar* d = dst.ptr<uchar>(i);
#         for (int j = start; j < cols; j++)
#         {
#             //如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
#             if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
#             {
#                 alpha = 1;
#             }
#             else
#             {
#                 //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好
#                 alpha = (processWidth - (j - start)) / processWidth;
#             }
#
#             d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
#             d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
#             d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
#
#         }
#     }
#
# }
import time
import numpy as np
import random


def func00():  # 生成随机数列表

    # random.seed(1)
    # kjz1=[random.randint(1,50) for j in range(0,7000)]
    # kjz1.extend([random.randint(80,150) for j in range(0,8000)])
    # kjz1.extend([random.randint(200,300) for j in range(0,5000)])
    # kjz1.extend([random.randint(400,500) for j in range(0,8000)])
    kjz1 = np.random.random(10)
    return kjz1


def func01(kjz1):  # 2分类

    bj = 1
    kjz1 = np.sort(kjz1)
    while (True):
        if bj == 1:
            kj = np.mean([kjz1[0], kjz1[len(kjz1) - 1]])  # 初始分组均值使用最小值和最大值的平均值
        else:
            k1 = s1
            k2 = s2
            kj = np.mean([k1, k2])
        kjz2 = [[], []]
        for j in kjz1:
            if j <= kj:
                kjz2[0].append(j)
            else:
                kjz2[1].append(j)
        s1 = np.mean(kjz2[0])
        s2 = np.mean(kjz2[1])
        if bj == 2:
            if s1 == k1 and s2 == k2:
                break
        bj = 2
    return kjz2


def func02(kjz1, k):  # k个均值分k份

    kjz1 = np.sort(kjz1)  # 正序
    wb2 = kjz1.copy()
    # 初始均匀分组wb1
    xlb = [];
    a = round(len(wb2) / (k));
    b = len(wb2) % (k)
    for j in range(1, k + 1):
        xlb.append(j * a)
        if j == k:
            xlb[j - 1] = xlb[j - 1] + b
    j = 0;
    wb1 = []
    for j in range(0, k):
        wb1.append([])
    i = 0;
    j = 0
    while (i <= len(wb2) - 1):
        wb1[j].append(wb2[i])
        if i >= xlb[j] - 1:
            j = j + 1
        i = i + 1
    kj1 = means(wb1)  # 初始分组均值

    bj = 1
    while (True):
        wb2 = kjz1.copy().tolist()
        if bj != 1:
            kj1 = kj2.copy()

        wb3 = []
        for j in range(0, k - 1):
            wb3.append([])
        for j in range(0, k - 1):
            i = -1
            while (True):
                if wb2[i] <= kj1[j]:
                    wb3[j].append(wb2.pop(i))
                else:
                    i = i + 1
                if i >= len(wb2):
                    break
        wb3.append(wb2)

        kj2 = means(wb3)  # 过程均值
        if bj == 2:
            if kj1 == kj2:
                break
        bj = 2
    return wb3


def means(lb1):  # 计算均值
    mean1 = [];
    mean2 = [];
    std1 = []
    for j in lb1:
        mean1.append(np.mean(j).tolist())
    for j in range(1, len(mean1)):
        mean2.append(np.mean([mean1[j - 1], mean1[j]]))  # 分组均值使用各组的均值
    print(mean2)
    return mean2


# Constants
INPUT_IMG_DIR = "./"
SIFT_OUT_IMG = "match.jpg"
RANSAC_OUT_IMG = "RANSAC.jpg"
STITCH_OUT_IMG = "merge.jpg"
STITCH_OUT_ALL_IMG = "Output_AllStitched.jpg"
RATIO_TEST_THRESOLD = 0.95  # 0.95 #0.77


class StitchInfo:
    def __init__(self):
        self.imgname = ''
        self.width = -1
        self.height = -1
        self.stitchindex = -1
        self.patch_left_points = None
        self.patch_right_points = None
        self.image = None
        self.patch_left_image = None
        self.patch_right_image = None
        self.match_patch = None
        self.match_points = None
        self.match_patch_step = 500
        self.shift_points = None


def trim(frame):
    # crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # crop bottom
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    # crop left
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    # crop right
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])

    return frame


def threshold_cluster(Data_set, threshold):
    # 统一格式化数据为一维数组
    stand_array = np.asarray(Data_set).ravel('C')
    stand_Data = Series(stand_array)
    index_list, class_k = [], []
    while stand_Data.any():
        if len(stand_Data) == 1:
            index_list.append(list(stand_Data.index))
            class_k.append(list(stand_Data))
            stand_Data = stand_Data.drop(stand_Data.index)
        else:
            class_data_index = stand_Data.index[0]
            class_data = stand_Data[class_data_index]
            stand_Data = stand_Data.drop(class_data_index)
            if (abs(stand_Data - class_data) <= threshold).any():
                args_data = stand_Data[abs(stand_Data - class_data) <= threshold]
                stand_Data = stand_Data.drop(args_data.index)
                index_list.append([class_data_index] + list(args_data.index))
                class_k.append([class_data] + list(args_data))
            else:
                index_list.append([class_data_index])
                class_k.append([class_data])
    return index_list, class_k


def get_sift_matches(img1, img2):
    """

    SIFT descriptor

    """
    i1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    i1 = histEqualize(i1)
    i2=histEqualize(i2)

    # Initiate SIFT detector
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.xfeatures2d.SURF_create()

    # Detects keypoints and computes the descriptors
    now =datetime.datetime.now()
    k1, des1 = sift.detectAndCompute(i1, None)
    k2, des2 = sift.detectAndCompute(i2, None)
    print("detect features cost:")
    print((datetime.datetime.now()-now).seconds)
    # BFMatcher with default params
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf = cv2.BFMatcher()
    # Finds the k best matches for each descriptor from a query set.
    # first one is queryDescriptor,second is trainDescriptor
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    final_matches = []
    img1h = img1.shape[0]
    img1w = img1.shape[1]
    img2h = img2.shape[0]
    img2w = img2.shape[1]
    for u, v in matches:
        # 匹配对过滤：
        # 从两个最近邻匹配对中做一个判断，若最近的匹配对小于第二个*ths，则保留这个匹配队。
        if u.distance < RATIO_TEST_THRESOLD * v.distance:
            final_matches.append([u])

    final_matches_filter = []
    anglelst = []

    for matchinfo in final_matches:
        pt1 = k1[matchinfo[0].queryIdx].pt  # 第一张图特征点坐标
        pt2 = k2[matchinfo[0].trainIdx].pt  # 第二张图特征点坐标

        # 计算距离：L2,第一张图的特征点在第二张图
        dis = math.sqrt((pt2[0] + img1w - pt1[0])**2 + (pt2[1] - pt1[1])**2)

        dy = pt2[1] - pt1[1]
        dx = pt2[0] + img1w - pt1[0]
        angle = math.atan(float(dy) / dx)
        pairtmp = (dis, angle)

        # if math.fabs(pt1[1]-pt2[1]) < img1h/60 and (img1w - pt1[0])<img1w/5 and pt2[0]< img2w/5:
        # if math.fabs(pt1[1] - pt2[1]) < img1h / 60 and (img1w - pt1[0]) < 400 and pt2[0] < 400:
        # 两个特征点的高度差小于图像高.. 并且 两个第一张图的特征点必须在右边，第二张图的特征点必须在左边
        if math.fabs(pt1[1] - pt2[1]) < img1h / 50 and (img1w - pt1[0]) < 300 and pt2[0] < 400:
            # if math.fabs(pt1[1] - pt2[1]) < img1h / 80 and math.fabs(angle) < 0.04 and math.fabs(angle) >0.03:
            final_matches_filter.append(matchinfo)
            anglelst.append(angle)
            # print(angle)
            # print(dis)
            # print(pt1,pt2)

    # cv2.drawMatchesKnn expects list of lists as matches.
    img_out = cv2.drawMatchesKnn(i1, k1, i2, k2, final_matches_filter, None, flags=2)  # final_matches
    anglenp = np.array(anglelst)
    # print(anglenp)
    # kjz2 = func01(anglenp)  # 2分类
    # for j in kjz2:
    #     print(j)
    #     print(len(j))
    # arr2 = kjz2[0]
    # kjz22 = func01(arr2)
    # for j in kjz22:
    #     print(j)
    #     print(len(j))
    # print("class over")
    cv2.imwrite(SIFT_OUT_IMG, img_out)
    cv2.namedWindow("SIFT Matches",cv2.WINDOW_NORMAL)
    cv2.imshow("SIFT Matches", img_out)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # flat_matches = np.asarray(final_matches_filter)
    return final_matches_filter, k1, k2


def concatImage(img1, img2, point1, point2):
    img1h = img1.shape[0]
    img1w = img1.shape[1]
    img2h = img2.shape[0]
    img2w = img2.shape[1]
    p1x = int(point1[0])
    p1y = int(point1[1])
    p2x = int(point2[0])
    p2y = int(point2[1])
    disx = p2x + img1w - p1x + 1
    img1crop = img1[:, :p1x]
    img1crop_same = img1[:, p1x:]
    img2crop_same = img2[:, 0:p2x]
    img2crop = img2[:, p2x:]
    img_create = np.zeros(img2crop.shape, np.uint8)
    img_create.fill(255)
    shifty = p2y - p1y
    if shifty <= 0:
        img2cropcrop = img2crop[:img2crop.shape[0] + shifty, :]
        img_create[0 - shifty:, :] = img2cropcrop
    else:
        img2cropcrop = img2crop[shifty:, :]
        img_create[0:img2crop.shape[0] - shifty, :] = img2cropcrop
    final = cv2.hconcat([img1crop, img_create])
    # kernel = np.ones((5, 5), np.float32) / 25
    # imgtmp = final[:, 1896 - 50:1896 + 100]
    # dst = cv2.filter2D(imgtmp, -1, kernel)
    # final[:, 1896 - 50:1896 + 100] = dst
    # dst = cv2.addWeighted(img1crop_same, 0.5, img2crop_same, 0.5, 0)
    return final


def concat_merge_Image(img1, img2, point1, point2):
    img1h = img1.shape[0]
    img1w = img1.shape[1]
    img2h = img2.shape[0]
    img2w = img2.shape[1]
    p1x = int(point1[0])
    p1y = int(point1[1])
    p2x = int(point2[0])
    p2y = int(point2[1])

    img1overlap = img1[:, p1x - p2x:]
    img1ori = img1[:, :p1x - p2x]

    img2_file = np.zeros(img2.shape, np.uint8)
    img2_file.fill(255)
    shifty = p2y - p1y
    if shifty <= 0:
        img2crop = img2[:img2h + shifty, :]
        img2_file[0 - shifty:, :] = img2crop
    else:
        img2crop = img2[shifty:, :]
        img2_file[:img2h - shifty, :] = img2crop
    img2overlap = img2_file[:, :p2x + img1w - p1x]
    img2ori = img2_file[:, p2x + img1w - p1x:]

    imgoh = img1overlap.shape[0]
    imgow = img1overlap.shape[1]
    imgoverlap = np.zeros(img1overlap.shape, np.uint8)
    imgoverlap.fill(255)
    for i in range(imgoh):
        for j in range(imgow):
            if img2overlap[i, j, 0] == 255 and img2overlap[i, j, 1] == 255 and img2overlap[i, j, 2] == 255:
                alpha = 1.0
            else:
                alpha = float(imgow - j) / imgow
            imgoverlap[i, j, 0] = int(img1overlap[i, j, 0] * alpha + img2overlap[i, j, 0] * (1.0 - alpha))
            imgoverlap[i, j, 1] = int(img1overlap[i, j, 1] * alpha + img2overlap[i, j, 1] * (1.0 - alpha))
            imgoverlap[i, j, 2] = int(img1overlap[i, j, 2] * alpha + img2overlap[i, j, 2] * (1.0 - alpha))
    final = cv2.hconcat([img1ori, imgoverlap, img2ori])

    return final


def getmatchImages(images):
    imgnum = len(images)
    patchs = []
    for i in range(imgnum):
        img = images[i]
        imgh = img.shape[0]
        imgw = img.shape[1]
        step = int(imgw / 5)
        if i == 0:
            patch = img[:, imgw - step:]
            patchs.append(patch)
        elif i == imgnum:
            path = img[:, :step]
        else:
            path = img[:, :step]
            patchs.append(path)
            patch = img[:, imgw - step:]
            patchs.append(patch)
    return path


# def getMatches():


def get_homography(img1, img2):
    """
    Get best merge
    """
    # ----------------------------------
    # 2. SIFT Operator22
    # ----------------------------------
    # find descriptors with SIFT and match keypoints
    matches, k1, k2 = get_sift_matches(img1, img2)
    if len(matches) > 0:
        flag = 0
        nums = len(matches)
        if nums < 3:
            index = 0
        else:
            index = int(nums / 2)

        for matchinfo in matches:
            pt1 = k1[matchinfo[0].queryIdx].pt  # 第二张图特征点坐标
            pt2 = k2[matchinfo[0].trainIdx].pt
            if flag == index:
                dst = concatImage(img1, img2, pt1, pt2)
                break
            flag += 1
    else:
        dst = cv2.hconcat([img1, img2])

    return trim(dst)
    # return dst


def get_homography_ori(img1, img2):
    """

    Get best homography matrix after running RANSAC algorithm

    """
    # ----------------------------------
    # 2. SIFT Operator22
    # ----------------------------------
    # find descriptors with SIFT and match keypoints
    matches, k1, k2 = get_sift_matches(img1, img2)

    # ----------------------------------
    # 3. RANSAC
    # ----------------------------------
    if len(matches[:, 0]) >= 4:
        src = np.float32([k1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        # H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
        H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)  # cv2.LMEDS

    else:
        raise AssertionError("Can't find enough keypoints.")

    dst = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    dst[0:img1.shape[0], 0:img1.shape[1]] = img1

    return trim(dst)


def stitch_N_images_ori(img_ls, num):
    """
    """
    cnt = 1
    out_img_idx = 4
    orig_len = len(img_ls)

    if num < 1:
        print("Need more than 1 image")
        return -1

    for idx in range(num - 1):
        img1 = img_ls.pop(0)
        img2 = img_ls.pop(0)

        # -------------
        #  Stitching
        # -------------
        op_img_12 = stitch_2_images(img1, img2, out_img_idx)

        out_img_idx = out_img_idx + 1
        img_ls.append(op_img_12)

    return op_img_12


def stitch_2_images(img1, img2, out_img_key):
    """

    Stitch and save image on to the disk.

    """
    out_img_name = str(out_img_key) + ".png"

    # ----------------------------------
    # 4. Stitch images
    # ----------------------------------
    print("Stitching images now. Please wait for a while ...")

    stitch_img = get_homography(img1, img2)

    print("Stitched image output:", out_img_name)
    stitch_img = np.uint8(stitch_img)
    cv2.imwrite(out_img_name, stitch_img)
    cv2.imshow("Stitched", stitch_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return stitch_img


def read_input_images(num):
    """

    Process user given images.

    """
    img_list = []

    for idx in range(num):
        # ---------------------------
        # ----- Read Images ---------
        # ---------------------------
        print("Enter image {} name: ".format(idx + 1))
        input_img_name = input()
        # Read image
        color_img = cv2.imread(INPUT_IMG_DIR + input_img_name)

        if color_img is None:
            print("Exception: Image not found: ", input_img_name)
            exit(1)

        img_list.append(color_img)

    # Stitch all images
    all_stitched = stitch_N_images(img_list, len(img_list))
    cv2.imwrite(STITCH_OUT_ALL_IMG, all_stitched)


def stitch_N_images(stitchImgs):
    """
    """
    num = len(stitchImgs)
    if num < 1:
        print("Need more than 1 image")
        return -1
    stictall = []
    newname = ''
    for idx in range(num - 1):
        stitch1 = stitchImgs[idx]
        stitch2 = stitchImgs[idx + 1]
        newname = stitchImgs[idx].imgname.split('.')[0] + stitchImgs[idx + 1].imgname.split('.')[0] + '.jpg'
        # -------------
        #  Stitching
        # -------------
        dst = None
        # img1 = stitch1.match_patch[1]
        # img2 = stitch2.match_patch[0]
        img1 = stitch1.image
        img2 = stitch2.image
        now = datetime.datetime.now()
        matches, k1, k2 = get_sift_matches(img1, img2)
        print("SIFT matches costs:")
        print((datetime.datetime.now()-now).seconds)
        checkmatch = []
        if len(matches) > 3:
            flag = 0

            # filter matches
            img1h = img1.shape[0]
            img1w = img1.shape[1]
            img2h = img2.shape[0]
            img2w = img2.shape[1]
            disdic = dict()
            angledic = dict()
            dislst = []
            anglelst = []
            index = 0
            for matchinfo in matches:
                pt1 = k1[matchinfo[0].queryIdx].pt  # 第一张图特征点坐标
                pt2 = k2[matchinfo[0].trainIdx].pt  # 第二张图特征点坐标
                dis = math.sqrt(
                    (pt2[0] + img1w - pt1[0]) * (pt2[0] + img1w - pt1[0]) + (pt2[1] - pt1[1]) * (pt2[1] - pt1[1]))
                dy = pt2[1] - pt1[1]
                dx = pt2[0] + img1w - pt1[0]
                angle = math.atan(float(dy) / dx)
                dislst.append(dis)
                if str(dis) not in disdic:
                    disdic[str(dis)] = index
                anglelst.append(angle)
                if str(angle) not in angledic:
                    angledic[str(angle)] = index
                index += 1
            disarr = np.array(dislst)
            anglearr = np.array(anglelst)
            kjz2 = func01(disarr)  # 2分类

            for j in kjz2:
                print(j)
                print(len(j))
            maxidx = -1
            if len(kjz2[0]) > len(kjz2[1]):
                maxidx = 0
            else:
                maxidx = 1
            angletmp = []
            for distmp in kjz2[maxidx]:
                angletmp.append(anglelst[disdic[str(distmp)]])
            angletmparr = np.array(angletmp)
            kjz2 = func01(angletmparr)  # 2分类
            for j in kjz2:
                print(j)
                print(len(j))
            maxidx = -1
            if len(kjz2[0]) > len(kjz2[1]):
                maxidx = 0
            else:
                maxidx = 1
            indexlst = []
            for angletmp in kjz2[maxidx]:
                indexlst.append(angledic[str(angletmp)])

            ##################
            # for matchinfo in matches:
            #     pt1 = k1[matchinfo[0].queryIdx].pt  # 第二张图特征点坐标
            #     pt2 = k2[matchinfo[0].trainIdx].pt
            nums = len(indexlst)
            if nums < 3:
                index_mid = 0
            else:
                index_mid = int(nums / 2)
            for indextmp in indexlst:
                matchinfo = matches[indextmp]
                pt1 = k1[matchinfo[0].queryIdx].pt  # 第一张图特征点坐标
                pt2 = k2[matchinfo[0].trainIdx].pt
                if flag == index_mid:
                    # dst = concatImage(img1, img2, pt1, pt2)
                    # pt1[0] = pt1[0]+ stitch1.width - stitch1.match_patch_step
                    stitchImgs[idx].match_points = (pt1, pt2)
                    checkmatch.append(matchinfo)
                    img_out = cv2.drawMatchesKnn(img1, k1, img2, k2, checkmatch, None, flags=2)  # final_matches
                    respath = './match_' + newname
                    cv2.imwrite(respath, img_out)
                    # test
                    # dst = concatImage(img1, img2, pt1, pt2)
                    dst = concat_merge_Image(img1, img2, pt1, pt2)
                    break
                flag += 1
        elif len(matches) > 0:
            # 后续需要加入策略，确定 0，1位置的
            matchinfo = matches[0]
            pt1 = k1[matchinfo[0].queryIdx].pt  # 第二张图特征点坐标
            pt2 = k2[matchinfo[0].trainIdx].pt
            stitchImgs[idx].match_points = (pt1, pt2)
            checkmatch.append(matchinfo)
            img_out = cv2.drawMatchesKnn(img1, k1, img2, k2, checkmatch, None, flags=2)  # final_matches
            respath = './res/xxxcheckmatch' + str(idx) + '.jpg'
            cv2.imwrite(respath, img_out)
            # test
            # dst = concatImage(img1, img2, pt1, pt2)
            dst = concat_merge_Image(img1, img2, pt1, pt2)
        else:
            dst = cv2.hconcat([img1, img2])

        stitchall = trim(dst)
        stitch_img = np.uint8(stitchall)
        cv2.imwrite('./res/' + newname, stitch_img)
        # stictall.append(dst)

    # for idx in range(len(stictall)-1):
    #     resall = cv2.hconcat([stictall[idx],stictall[idx+1]])
    #     stitchall =  trim(resall)
    #     stitch_img = np.uint8(stitchall)
    #     cv2.imwrite('./stitch_test_'+str(idx)+'.jpg',stitch_img)
    return 0


def getImglst():
    imgpathlists = []
    imgpath = 'D:/stitch_test/1/1_Line17_up_20190411032435_1_34km+774.8m_forward.jpg'
    imgpathlists.append(imgpath)
    imgpath = 'D:/stitch_test/2/2_Line17_up_20190411032435_1_34km+775.0m_forward.jpg'
    imgpathlists.append(imgpath)
    imgpath = 'D:/stitch_test/3/3_Line17_up_20190411032435_1_34km+775.0m_forward.jpg'
    imgpathlists.append(imgpath)
    imgpath = 'D:/stitch_test/4/4_Line17_up_20190411032435_1_34km+775.0m_forward.jpg'
    imgpathlists.append(imgpath)
    imgpath = 'D:/stitch_test/5/5_Line17_up_20190411032435_1_34km+774.9m_forward.jpg'
    imgpathlists.append(imgpath)
    imgpath = 'D:/stitch_test/6/6_Line17_up_20190411032435_1_34km+774.7m_forward.jpg'
    imgpathlists.append(imgpath)
    imgpath = 'D:/stitch_test/7/7_Line17_up_20190411032435_1_34km+774.7m_forward.jpg'
    imgpathlists.append(imgpath)
    index = 1
    stitchImgs = []
    for imagepath in imgpathlists:
        name = os.path.basename(imagepath)
        img = cv2.imread(imagepath)
        imgh = img.shape[0]
        imgw = img.shape[1]
        stitchInfo = StitchInfo()
        stitchInfo.imgname = name
        stitchInfo.width = imgw
        stitchInfo.height = imgh
        stitchInfo.image = img
        stitchInfo.stitchindex = index
        stitchInfo.match_patch = []
        patch1 = stitchInfo.image[:, :stitchInfo.match_patch_step]
        stitchInfo.match_patch.append(patch1)
        patch2 = stitchInfo.image[:, imgw - stitchInfo.match_patch_step:]
        stitchInfo.match_patch.append(patch2)
        index += 1
        stitchImgs.append(stitchInfo)
    return stitchImgs


def stitch_images():
    stitchImgs = getImglst()
    all_stitched = stitch_N_images(stitchImgs)


# Main
if __name__ == "__main__":
    stitch_images()
