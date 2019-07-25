import math
import cv2
import numpy as np
from ..misc import cluster
# from .merge import concat_merge_Image
from .merge_image import concat_merge_Image

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


def merge_image(img1, img2,
                matches, k1, k2,
                img1_src=None, img2_src=None,
                default_overlap_ratio=0.20,
                out_match=None, ret_image=True,
                stitch_src=False, restore_size=False):
    """
    merge two images given matches and keypoints.
    :param default_overlap_ratio: if not matches are founded,
    then we set a default overlap ratio for each image,and
    use this default value to merge image.
    :param out_match:If output matched results.
    """
    bgr = True if len(img1.shape) == 3 else False
    if stitch_src:
        assert img1_src is not None and img2_src is not None
        bgr = True if len(img1_src.shape) == 3 else False

    img1h, img1w = img1.shape[0], img1.shape[1]
    img1_src_h, img1_src_w = img1_src.shape[0], img1_src.shape[1]
    img2_src_h, img2_src_w = img2_src.shape[0], img2_src.shape[1]
    img2h, img2w = img2.shape[0], img2.shape[1]
    resize = img1h / img1_src_h
    img1_resize = img1_src.copy()
    img2_resize = img2_src.copy()
    if not img1_src_h == img1h or not img1_src_w == img1w:
        img1_resize = cv2.resize(img1_src, (img1w, img1h))
        img2_resize = cv2.resize(img2_src, (img2w, img2h))

    draw_match = []
    if len(matches) > 3:
        flag = 0
        # filter matches
        # compute distance of coordinate and angles for each matches
        disdic, angledic, dislst, anglelst = compute_dist_angle(matches, k1, k2, img1w)
        disarr = np.array(dislst)
        kjz2 = cluster(disarr)  # 对距离做一个二分类（均值作为条件）
        indexlst = get_match_index(kjz2, disdic, angledic, anglelst)  # 对角度做分类找到最合适的match索引
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
                if stitch_src and not restore_size:
                    dst, img1ori, img2ori, imgoverlap, shifty = concat_merge_Image(img1_resize, img2_resize,
                                                                                   match_points[0],
                                                                                   match_points[1], bgr=bgr)
                elif restore_size:
                    pt1 = (pt1[0] / resize, pt1[1] / resize)
                    pt2 = (pt2[0] / resize, pt2[1] / resize)
                    match_points = (pt1, pt2)
                    dst, img1ori, img2ori, imgoverlap, shifty = concat_merge_Image(img1_src, img2_src, match_points[0],
                                                                                   match_points[1],
                                                                                   bgr=bgr)
                else:
                    dst, img1ori, img2ori, imgoverlap, shifty = concat_merge_Image(img1, img2, match_points[0],
                                                                                   match_points[1], bgr=bgr)
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
        if stitch_src and not restore_size:
            dst, img1ori, img2ori, imgoverlap, shifty = concat_merge_Image(img1_resize, img2_resize, match_points[0],
                                                                           match_points[1],
                                                                           bgr=bgr)
        elif restore_size:
            pt1 = (pt1[0] / resize, pt1[1] / resize)
            pt2 = (pt2[0] / resize, pt2[1] / resize)
            match_points = (pt1, pt2)
            dst, img1ori, img2ori, imgoverlap, shifty = concat_merge_Image(img1_src, img2_src, match_points[0],
                                                                           match_points[1],
                                                                           bgr=bgr)
        else:
            dst, img1ori, img2ori, imgoverlap, shifty = concat_merge_Image(img1, img2, match_points[0], match_points[1],
                                                                           bgr=bgr)
        print("match")
    else:
        if stitch_src and not restore_size:
            img1ori = img1_src[:, :img1w - int(img1w * default_overlap_ratio)]
            img2ori = img2_src[:, int(img2w * default_overlap_ratio):]
            imgoverlap = img2_src[:, :int(img2w * default_overlap_ratio)] * 0.5 + img1_src[:, img1w - int(
                img1w * default_overlap_ratio):] * 0.5
            imgoverlap = np.array(imgoverlap, dtype=np.uint8)
        elif stitch_src and restore_size:
            img1ori = img1_src[:, :img1_src_w - int(img1_src_w * default_overlap_ratio)]
            img2ori = img2_src[:, int(img2_src_w * default_overlap_ratio):]
            imgoverlap = img2_src[:, :int(img2_src_w * default_overlap_ratio)] * 0.5 + img1_src[:, img1_src_w - int(
                img1_src_w * default_overlap_ratio):] * 0.5
            imgoverlap = np.array(imgoverlap, dtype=np.uint8)
        else:
            img1ori = img1[:, :img1w - int(img1w * default_overlap_ratio)]
            img2ori = img2[:, int(img2w * default_overlap_ratio):]
            imgoverlap = img2[:, :int(img2w * default_overlap_ratio)] * 0.5 + img1[:, img1w - int(
                img1w * default_overlap_ratio):] * 0.5
            imgoverlap = np.array(imgoverlap, dtype=np.uint8)
        shifty = 0
        dst = cv2.hconcat([img1ori, imgoverlap, img2ori])
        print("not match")

    # draw best match
    if out_match is not None:
        img_out = cv2.drawMatchesKnn(img1, k1, img2, k2, draw_match, None, flags=2)
        if img_out is not None:
            respath = out_match
            cv2.imwrite(respath, img_out)

    stitchall = crop_zero(dst)  # remove zero rows and columns
    stitch_img = np.uint8(stitchall)

    if ret_image:
        return stitch_img, img1ori, img2ori, imgoverlap, shifty
    else:
        return img1ori, img2ori, imgoverlap, shifty
