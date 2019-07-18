import math
import cv2
import numpy as np
from ..misc import cluster


def crop_zero(frame):
    # crop top
    if not np.sum(frame[0]):
        return crop_zero(frame[1:])
    # crop bottom
    if not np.sum(frame[-1]):
        return crop_zero(frame[:-2])
    # crop left
    if not np.sum(frame[:, 0]):
        return crop_zero(frame[:, 1:])
    # crop right
    if not np.sum(frame[:, -1]):
        return crop_zero(frame[:, :-2])
    return frame


def compute_dist_angle(matches, k1, k2, img_w):
    disdic = dict()
    angledic = dict()
    dislst = []
    anglelst = []
    index = 0
    for matchinfo in matches:
        pt1 = k1[matchinfo[0].queryIdx].pt  # 第一张图特征点坐标
        pt2 = k2[matchinfo[0].trainIdx].pt  # 第二张图特征点坐标
        dis = math.sqrt(
            (pt2[0] + img_w - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        dy = pt2[1] - pt1[1]
        dx = pt2[0] + img_w - pt1[0]
        angle = math.atan(float(dy) / dx)
        dislst.append(dis)
        if str(dis) not in disdic:
            disdic[str(dis)] = index
        anglelst.append(angle)
        if str(angle) not in angledic:
            angledic[str(angle)] = index
        index += 1
    return disdic, angledic, dislst, anglelst

def get_match_index(kjz2, disdic, angledic, anglelst):
    if len(kjz2[0]) > len(kjz2[1]):
        maxidx = 0
    else:
        maxidx = 1
    angletmp = []
    for distmp in kjz2[maxidx]:
        angletmp.append(anglelst[disdic[str(distmp)]])
    angletmparr = np.array(angletmp)
    kjz2 = cluster(angletmparr)  # 2分类

    if len(kjz2[0]) > len(kjz2[1]):
        maxidx = 0
    else:
        maxidx = 1
    indexlst = []
    for angletmp in kjz2[maxidx]:
        indexlst.append(angledic[str(angletmp)])
    return indexlst


def concat_merge_Image(img1, img2, point1, point2,
                       bgr=False):
    """
    concat and merge images,
    :param bgr: (boolean) if True,concat brg
    :return:stitched image,img1ori,img2ori,imgoverlap
    """
    img1h, img1w = img1.shape[0], img1.shape[1]
    img2h, img2w = img2.shape[0], img2.shape[1]
    p1x, p1y = int(point1[0]), int(point1[1])
    p2x, p2y = int(point2[0]), int(point2[1])

    img1overlap = img1[:, p1x - p2x:]
    img1ori = img1[:, :p1x - p2x]

    img2_file = np.zeros(img2.shape, np.uint8)
    # img2_file.fill(255)
    shifty = p2y - p1y  # 若右侧图片的keypoints在左侧上方，对图片上面部分做裁剪，否则对图片下面部分做裁剪，空出部分填充0
    if shifty <= 0:
        img2crop = img2[:img2h + shifty, :] # 裁剪右图，使其与左图对齐
        img2_file[0 - shifty:, :] = img2crop
    else:
        img2crop = img2[shifty:, :]
        img2_file[:img2h - shifty, :] = img2crop

    img2overlap = img2_file[:, :p2x + img1w - p1x]
    img2ori = img2_file[:, p2x + img1w - p1x:]

    imgoh = img1overlap.shape[0]
    imgow = img1overlap.shape[1]
    imgoverlap = np.zeros(img1overlap.shape, np.uint8)
    # imgoverlap.fill(255)
    # BRG图像拼接
    if bgr:
        for i in range(imgoh):
            for j in range(imgow):
                if img2overlap[i, j, 0] == 0 and img2overlap[i, j, 1] == 0 and img2overlap[i, j, 2] == 0:
                    alpha = 1.0
                else:
                    alpha = float(imgow - j) / imgow
                imgoverlap[i, j, :] = img1overlap[i, j, :] * alpha + img2overlap[i, j, :] * (1.0 - alpha)

    else:   # 灰度图像拼接
        for i in range(imgoh):
            for j in range(imgow):
                if img2overlap[i, j] == 0:
                    alpha = 1.0
                else:
                    alpha = float(imgow - j) / imgow
                imgoverlap[i, j] = int(img1overlap[i, j] * alpha + img2overlap[i, j] * (1.0 - alpha))
    final = cv2.hconcat([img1ori, imgoverlap, img2ori])
    return final,img1ori, img2ori, imgoverlap,shifty

def merge_image(img1, img2,
                matches, k1, k2,
                img1_src=None,img2_src=None,
                default_overlap_ratio=0.20,
                out_match=None,ret_image=True,
                stitch_src=False):
    """
    merge two images given matches and keypoints.
    :param default_overlap_ratio: if not matches are founded,
    then we set a default overlap ratio for each image,and
    use this default value to merge image.
    :param out_match:If output matched results.
    """
    bgr=True if len(img1.shape)==3 else False
    if stitch_src:
        assert img1_src is not None and img2_src is not None
        bgr=True if len(img1_src.shape)==3 else False

    img1h, img1w = img1.shape[0], img1.shape[1]
    img1_src_h,img1_src_w=img1_src.shape[0],img1_src.shape[1]
    img2h, img2w = img2.shape[0], img2.shape[1]
    if not img1_src_h == img1h or not img1_src_w ==img1w:
        img1_src=cv2.resize(img1_src,(img1w,img1h))
        img2_src=cv2.resize(img2_src,(img2w,img2h))

    draw_match=[]
    if len(matches) > 3:
        flag = 0
        # filter matches
        # compute distance of coordinate and angles for each matches
        disdic, angledic, dislst, anglelst = compute_dist_angle(matches, k1, k2, img1w)
        disarr = np.array(dislst)
        kjz2 = cluster(disarr)  # 对距离做一个二分类（均值作为条件）
        indexlst = get_match_index(kjz2, disdic, angledic, anglelst) # 对角度做分类找到最合适的match索引
        # for matchinfo in matches:
        nums = len(indexlst)
        index_mid = 0 if nums < 3 else int(nums / 2)
        for indextmp in indexlst:
            matchinfo = matches[indextmp]
            pt1 = k1[matchinfo[0].queryIdx].pt  # 第一张图特征点坐标
            pt2 = k2[matchinfo[0].trainIdx].pt
            if flag == index_mid:
                match_points = (pt1, pt2)
                draw_match.append(matchinfo)
                if stitch_src:
                    dst, img1ori, img2ori, imgoverlap, shifty = concat_merge_Image(img1_src, img2_src, match_points[0],
                                                                                   match_points[1], bgr=bgr)
                else:
                    dst, img1ori, img2ori, imgoverlap,shifty = concat_merge_Image(img1, img2, match_points[0],
                                                                              match_points[1],bgr=bgr)
                print("match")
                break
            flag += 1
    elif len(matches) > 0:
        # 后续需要加入策略，确定 0，1位置的
        matchinfo = matches[0]
        pt1 = k1[matchinfo[0].queryIdx].pt  # 第一张图特征点坐标
        pt2 = k2[matchinfo[0].trainIdx].pt
        match_points = (pt1, pt2)
        draw_match.append(matchinfo)
        if stitch_src:
            dst, img1ori, img2ori, imgoverlap, shifty = concat_merge_Image(img1_src, img2_src, match_points[0], match_points[1],
                                                                           bgr=bgr)
        else:
            dst,img1ori,img2ori,imgoverlap,shifty = concat_merge_Image(img1, img2, match_points[0], match_points[1],bgr=bgr)
        print("match")
    else:
        if stitch_src:
            img1ori = img1_src[:, :img1w - int(img1w * default_overlap_ratio)]
            img2ori = img2_src[:, int(img2w * default_overlap_ratio):]
            imgoverlap = img2_src[:, :int(img2w * default_overlap_ratio)] * 0.5 + img1_src[:, img1w - int(
                img1w * default_overlap_ratio):] * 0.5
            imgoverlap = np.array(imgoverlap, dtype=np.uint8)
        else:
            img1ori=img1[:,:img1w-int(img1w*default_overlap_ratio)]
            img2ori=img2[:,int(img2w*default_overlap_ratio):]
            imgoverlap=img2[:,:int(img2w*default_overlap_ratio)] * 0.5 +img1[:,img1w-int(img1w*default_overlap_ratio):]*0.5
            imgoverlap=np.array(imgoverlap,dtype=np.uint8)
        shifty=0
        dst = cv2.hconcat([img1ori,imgoverlap,img2ori])
        print("not match")

    # draw best match
    if out_match is not None:
        img_out = cv2.drawMatchesKnn(img1, k1, img2, k2,draw_match, None, flags=2)
        if img_out is not None:
            respath = out_match
            cv2.imwrite(respath, img_out)

    stitchall = crop_zero(dst)  # remove zero rows and columns
    stitch_img = np.uint8(stitchall)

    if ret_image:
        return stitch_img,img1ori,img2ori,imgoverlap,shifty
    else:
        return img1ori,img2ori,imgoverlap,shifty

