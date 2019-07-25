import numpy as np
import cv2
from .misc import shift_bboxes_to_stitch, reshape_bboxes
from mtcv.bbox_ops import merge_bbox,iou,bbox_equa

def bbox_to_patch(bboxes,  # bboxes
                  shiftys,  # sfhitys
                  imgs_left,  # imgori1
                  imgs_overlap,  # overlap region
                  imgs_right,  # imgori2
                  imgs_stitch,  # a set of stitched img patches
                  images_w):  # normal image width.
    """
    transform bbox of single image to bbox of stitching patches.
    :param bboxes: (None,5)
    :param shiftys: (shift of each images)
    :param imgs_left:
    :param imgs_overlap:
    :param imgs_right:
    :param imgs_stitch:
    :param images_w:
    :return:
    """
    bboxes_left = []
    bboxes_right = []
    bboxes_mid = []
    bboxes_left_ovr = []
    bboxes_right_ovr = []
    for i in range(1, len(imgs_overlap)):
        bboxes_ovr = []  # used to keep bboxes in overlap part.
        bboxes_right_img = bboxes[i]  # get bboxes of img right.
        bbox_shift = shiftys[:i].sum()  # if shift <=0,means right img above left,need move down.
        if i == 1:
            bboxes_left_img = bboxes[i - 1]  # get bboxes of img left.
            left_w = imgs_left[0].shape[1]
            if len(bboxes_left_img) < 1:
                pass
            else:
                for bbox in bboxes_left_img:
                    xmin, ymin, xmax, ymax, score = bbox
                    ovr_ymin = ymin - bbox_shift
                    ovr_ymax = ymax - bbox_shift
                    # bbox completely not in stitch region.Just keep it .
                    if xmax <= left_w and xmin < left_w:
                        bboxes_left.append(bbox)
                    # part of bbox in seam and other not.
                    elif xmax > left_w and xmin < left_w:
                        left_xmax = left_w
                        bboxes_left.append([xmin, ymin, left_xmax, ymax, score])
                        ovr_xmin = 0
                        ovr_xmax = xmax - left_w
                        bboxes_ovr.append([ovr_xmin, ovr_ymin, ovr_xmax, ovr_ymax, score])
                    # bbox in overlap region.
                    elif xmax > left_w and xmin >= left_w:
                        ovr_xmin = xmin - left_w
                        ovr_xmax = xmax - left_w
                        bboxes_ovr.append([ovr_xmin, ovr_ymin, ovr_xmax, ovr_ymax, score])

        bboxes_mid_tmp = []  # used to keep bboxes in right part.
        overlap_w = imgs_overlap[i - 1].shape[1]  # get overlap width, to compute relative coordinate of bbox.
        overlap_w2 = imgs_overlap[i].shape[1]
        right_w = imgs_right[i - 1].shape[1]
        bboxes_ovr2 = []  # overlap region of right part.
        if len(bboxes_right_img) < 1:
            pass
        else:
            for bbox in bboxes_right_img:
                xmin, ymin, xmax, ymax, score = bbox
                ymin = ymin - bbox_shift
                ymax = ymax - bbox_shift
                # bbox in left overlap region.
                if xmin < overlap_w and xmax <= overlap_w:
                    bboxes_ovr.append([xmin, ymin, xmax, ymax, score])
                # part of bbox in left overlap region and other not.
                elif xmin < overlap_w and xmax > overlap_w:
                    bboxes_ovr.append([xmin, ymin, overlap_w, ymax, score])
                    # bbox in middle part.
                    if xmax <= images_w - overlap_w2:
                        xmin_mid = 0
                        xmax_mid = xmax - overlap_w
                        bboxes_mid_tmp.append([xmin_mid, ymin, xmax_mid, ymax, score])
                    # bbox in right overlap region.
                    else:
                        xmin_mid = 0
                        xmax_mid = right_w - overlap_w2
                        bboxes_mid_tmp.append([xmin_mid, ymin, xmax_mid, ymax, score])
                        xmin_right = 0
                        xmax_right = xmax - (images_w - overlap_w2)
                        bboxes_ovr2.append([xmin_right, ymin, xmax_right, ymax, score])
                # bbox in middle region.
                elif xmin >= overlap_w and xmax <= images_w - overlap_w2:
                    xmin_mid = xmin - overlap_w
                    xmax_mid = xmax - overlap_w
                    bboxes_mid_tmp.append([xmin_mid, ymin, xmax_mid, ymax, score])
                # part of bbox in middle region,other in right overlap.
                elif xmin >= overlap_w and xmax > images_w - overlap_w2:
                    if xmin < images_w - overlap_w2:
                        xmin_mid = xmin - overlap_w
                        xmax_mid = right_w - overlap_w2
                        bboxes_mid_tmp.append([xmin_mid, ymin, xmax_mid, ymax, score])
                        xmin_right = 0
                        xmax_right = xmax - (images_w - overlap_w2)
                        bboxes_ovr2.append([xmin_right, ymin, xmax_right, ymax, score])
                    elif xmin >= images_w - overlap_w2:
                        xmin_right = xmin - (images_w - overlap_w2)
                        xmax_right = xmax - (images_w - overlap_w2)
                        bboxes_ovr2.append([xmin_right, ymin, xmax_right, ymax, score])
        if i == (len(imgs_overlap) - 1):
            bboxes_right_img = bboxes[i + 1]  # get last img bbox
            bbox_shift = shiftys.sum()
            if len(bboxes_right_img) < 1:
                pass
            else:
                for bbox in bboxes_right_img:
                    xmin, ymin, xmax, ymax, score = bbox
                    ymin = ymin - bbox_shift
                    ymax = ymax - bbox_shift
                    if xmin < overlap_w2 and xmax <= overlap_w2:
                        bboxes_ovr2.append([xmin, ymin, xmax, ymax, score])
                    elif xmin < overlap_w2 and xmax > overlap_w2:
                        bboxes_ovr2.append([xmin, ymin, overlap_w2, ymax, score])
                        xmin_right = 0
                        xmax_right = xmax - overlap_w2
                        bboxes_right.append([xmin_right, ymin, xmax_right, ymax, score])
                    elif xmin > overlap_w2 and xmax > overlap_w2:
                        xmin_right = xmin - overlap_w2
                        xmax_right = xmax - overlap_w2
                        bboxes_right.append([xmin_right, ymin, xmax_right, ymax, score])
        bboxes_mid.append(bboxes_mid_tmp)
        bboxes_left_ovr.append(bboxes_ovr)
        bboxes_right_ovr.append(bboxes_ovr2)

    w = []
    for i in range(len(imgs_stitch)):
        w_tmp = imgs_stitch[i].shape[1]
        w += [w_tmp]
    w = np.array(w)
    for i in range(len(bboxes_mid)):
        bboxes_left_ovr[i] = shift_bboxes_to_stitch(bboxes_left_ovr[i], w[:2 * i + 1].sum())
        bboxes_mid[i] = shift_bboxes_to_stitch(bboxes_mid[i], w[:2 * i + 2].sum())
        bboxes_right_ovr[i] = shift_bboxes_to_stitch(bboxes_right_ovr[i], w[:2 * i + 3].sum())
        if i == (len(bboxes_mid) - 1):
            bboxes_right = shift_bboxes_to_stitch(bboxes_right, w[:-1].sum())
    bboxes_left_ovr = reshape_bboxes(bboxes_left_ovr)
    bboxes_mid = reshape_bboxes(bboxes_mid)
    bboxes_right_ovr = reshape_bboxes(bboxes_right_ovr)
    bboxes_all = bboxes_left + bboxes_left_ovr + bboxes_mid + bboxes_right_ovr + bboxes_right
    return bboxes_all


# def concat_bbox(bboxes):

def imgs_to_patches(imgs_left, imgs_overlap, imgs_right, shiftys):
    # keep first img and last img
    img_first = imgs_left[0]
    img_last = imgs_right[-1]
    imgs_stitch = []
    for i in range(1, len(imgs_overlap)):
        if i == 1:
            imgs_stitch.append(img_first)
            imgs_stitch.append(imgs_overlap[0])
        # all we need are current left img, and previous width of overlap.
        # then get left[:,width:],are middle img preserved to stitch.
        # since there may be shift in different images,so we need to
        # refine them.
        img_overlap = imgs_overlap[i]
        ovr_cur_w, ovr_cur_h = img_overlap.shape[1], img_overlap.shape[0]

        shifty = shiftys[:i].sum()

        img_right = imgs_right[i - 1]
        img_w, img_h = img_right.shape[1], img_right.shape[0]

        right_tmp = np.zeros(img_right.shape, np.uint8)
        ovr_tmp = np.zeros(imgs_overlap[i].shape, np.uint8)
        if shifty <= 0:
            ovrcrop = img_overlap[:ovr_cur_h + shifty, :]  # 裁剪右图，使其与左图对齐
            ovr_tmp[0 - shifty:, :] = ovrcrop
        else:
            ovrcrop = img_overlap[shifty:, :]
            ovr_tmp[:ovr_cur_h - shifty, :] = ovrcrop

        # 从i>1开始，需要对右边的图片进行偏移矫正
        if i > 1:
            shifty = shiftys[:i - 1].sum()
            if shifty <= 0:
                rightcrop = img_right[:img_h + shifty, :]
                right_tmp[0 - shifty:, :] = rightcrop
            else:
                rightcrop = img_right[shifty:, :]
                right_tmp[:img_h - shifty, :] = rightcrop

        if i == 1:
            img_mid = img_right[:, :img_right.shape[1] - ovr_cur_w]
        else:
            img_mid = right_tmp[:, :img_right.shape[1] - ovr_cur_w]

        imgs_stitch.append(img_mid)
        imgs_stitch.append(ovr_tmp)
        # 对最后的一张图做偏移矫正，并加到拼接列表
        if i == len(imgs_overlap) - 1:
            img_last_keep = np.zeros(img_last.shape, np.uint8)
            img_right_w, img_right_h = img_last.shape[1], img_last.shape[0]
            shifty = shiftys.sum()
            if shifty <= 0:
                last_tmp = img_last[:img_right_h + shifty, :]  # 裁剪右图，使其与左图对齐
                img_last_keep[0 - shifty:, :] = last_tmp
            else:
                last_tmp = img_last[shifty:, :]
                img_last_keep[:ovr_cur_h - shifty, :] = last_tmp
            imgs_stitch.append(img_last_keep)
    return imgs_stitch


def concat_bbox(bboxes):
    """
    concat two neighboured bboxes in horizontal direction.
    :param bboxes:
    :return:
    """
    count = 0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes) - 1):
            bbox = merge_bbox(bboxes[i], bboxes[j])
            if bbox is not None:
                count += 1
                bboxes[i] = bbox
                bboxes[j] = bbox
    i = 0
    while True:
        j = i + 1
        while True:
            if bbox_equa(bboxes[i], bboxes[j]):
                del bboxes[j]
            else:
                j += 1
            if j >= len(bboxes):
                break
        i += 1
        if i == len(bboxes) - 1:
            break
    return bboxes



