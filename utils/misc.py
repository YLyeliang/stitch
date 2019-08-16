import numpy as np
from mtcv import bbox_to_patch,concat_bbox,nms
import cv2

def draw_bbox(bboxes, shiftys, imgs_left_shape, imgs_overlap_shape, imgs_right_shape,
              coordinates, images_w):
    # stitched = cv2.imread(stitched_img_path)
    bboxes_all = bbox_to_patch(bboxes, shiftys, imgs_left_shape, imgs_overlap_shape, imgs_right_shape,
                               coordinates, images_w)
    bboxes_all = concat_bbox(bboxes_all)
    bboxes_all, inds = nms(bboxes_all)
    # stitched = draw_bboxes(stitched, bboxes_all)
    return bboxes_all


def get_patch_shape(imgs_stitch, imgs_left, imgs_overlap, imgs_right):
    patches_width = [patch.shape[1] for patch in imgs_stitch]
    patches_width = np.array(patches_width)  # patches num: 2*i-1
    coordinates = [patches_width[:i + 1].sum() for i in range(len(patches_width))]
    imgs_left_shape = [part.shape for part in imgs_left]
    imgs_overlap_shape = [part.shape for part in imgs_overlap]
    imgs_right_shape = [part.shape for part in imgs_right]
    return coordinates, imgs_left_shape, imgs_overlap_shape, imgs_right_shape


def draw_segment(stitched_img,stitched_name,segments_info,concat_width=500,x_width=20):
    """
    Add segment information on the right side of image.
    segment_info should provided,thus which section image are loacted at
    can be calculated.
    :param segment_info: (list(list))a list containing segment information,
    :return:
    """
    # transfer image miles str to numbers
    miles = stitched_name.split('_')[4]
    km, m = miles.split('+')
    km = int(km[:-2])
    m = float(m[:-1])
    mile = km * 1000 + m
    assert len(segments_info[0])>1,'segments info should have shape list(list) e.g [[1,2,3]] ,not [1,2,3]'
    for segment_info in segments_info:
        assert len(segment_info) == 5, "segment information should contain orderly:mile_start,mile_end," \
                                       "segment_num,segment_start,segmeng_end."
        mile_start,mile_end,segment_num,segment_start,segment_end=segment_info
        distance = abs(mile_start - mile_end)
        if mile_start < mile_end:
            mile_to_start = mile - mile_start
            mile_to_end = mile_end - mile
        else:
            mile_to_start = mile_start - mile
            mile_to_end = mile - mile_end
        if mile_to_start > -10 and mile_to_end > 0:
            concat_segment(stitched_img,distance,mile_to_start,segment_num,segment_start,segment_end,concat_width,x_width)

def concat_segment(stitched_img,distance, mile_to_start,segment_num, segment_start,
                   segment_end, concat_width, x_width):
    """
    Add segment information on the right side of image.
    mile start and end should specified,thus which section image are loacted at
    can be calculated.
    :return:
    """

    img_h = stitched_img.shape[0]
    segment_width = distance / segment_num  # segment width, e.g 1.2m
    seg_num_img_int = round(10 / segment_width)
    seg_num_img = 10 / segment_width  # number of segment on single image
    seg_pix_integral = round(img_h / seg_num_img)

    # calculate index of segment,
    # get y coordinates of all segment
    order = segment_start < segment_end
    y_cord = [y * seg_pix_integral for y in range(seg_num_img_int)]
    seg_num_div = int(mile_to_start // 10)  # 计算距里程起点经过的管片数量，单张图片管片数*这个数
    seg_num_mod = round(mile_to_start % 10)
    seg_num_pass = round(seg_num_div * seg_num_img) + round(seg_num_mod / segment_width)
    if order:
        segment_id = [seg_num_pass + i for i in range(len(y_cord))]
    else:
        segment_id = [seg_num_pass - i for i in range(len(y_cord))]

    # if mile_to_start < 0,
    # which means that first segment showed up.else occupied segments.
    if mile_to_start < 0:
        seg_num_img = round((10 + mile_to_start) / segment_width)
        start_y = round(abs(mile_to_start) / 10 * 4000)
        y_cord = [start_y + y * seg_pix_integral for y in range(seg_num_img)]
        if order:
            segment_id = [segment_start + i for i in range(len(y_cord))]
        else:
            segment_id = [segment_start - i for i in range(len(y_cord))]

    bgr = np.zeros((img_h, concat_width, 3), dtype=np.uint8)
    for i in range(len(segment_id)):
        cv2.putText(bgr, "#{}".format(segment_id[i]), (x_width, 300 + y_cord[i]),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 5,
                    color=(0, 0, 255), thickness=3)
    stitched_img = cv2.hconcat([stitched_img, bgr])
    return stitched_img,segment_id



def cluster(kjz1):
    """
    perform a means 2 cluster. e.g 2-cluster approach,
    Compute a mean of whole data.assign a class smaller than mean,while other class greater.
    Recompute means of each classes,Taken the mean of both class as new mean,and redo it
    until it not changed.
    :param kjz1:
    :return:
    """

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

