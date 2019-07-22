import cv2
import numpy as np
import os
from mtcv import bgr2gray, histEqualize
from mtcv import read_txt_mklist
from utils.merge import merge_image
from utils.features import compute_matches
import math


class Stitcher(object):
    """
    A stitching class used for image stitching.
    It consists of following modules:
    img_resize: resize img by given ratio or dsize.
    preprocess:perform preprocess orderly by resize,bgr2gray,histogram equalization.

    """

    def __init__(self, resize=0.5,
                 is_norm=True,
                 is_2gray=True,
                 descriptor_type='sift',
                 match_type='knn',
                 show=True):
        self.resize = resize
        self.descriptor_type = descriptor_type
        self.match_type = match_type
        self.is_norm = is_norm
        self.is_2gray = is_2gray
        self.show = show

    def img_resize(self, img, ratio=0.5, dsize=None):
        """
        Resize single image to desired size.
        :param dsize: desired size.
        :param ratio: If given,image will be resize proportional to ratio.
        :return:
        """
        img_h, img_w = img.shape[0], img.shape[1]
        self.img_h = img_h
        self.img_w = img_w
        if dsize is not None:
            self.img_resize_h = dsize[1]
            self.img_resize_w = dsize[0]
            return cv2.resize(img, (dsize[0], dsize[1]))
        else:
            new_h, new_w = round(img_h * ratio), round(img_w * ratio)
            self.img_resize_h = new_h
            self.img_resize_w = new_w
            img = cv2.resize(img, (new_w, new_h))
            return img

    def preprocess(self, images, is_2gray=True, resize_ratio=None, is_norm=False):
        """
        Preprocess a set of images,including resize,bgr2gray, histogram equalization.
        :param resize_ratio:(float) resizing ratio.
        """
        imgs = images.copy()
        if not isinstance(imgs[0], str):
            if is_2gray:
                for i, img in enumerate(imgs):
                    imgs[i] = bgr2gray(img)
            if resize_ratio is not None:
                for i, img in enumerate(imgs):
                    imgs[i] = self.img_resize(img, resize_ratio)
            if is_norm:
                for i, img in enumerate(imgs):
                    imgs[i] = histEqualize(img, 'clahe')
        # else what we read is string containing img paths.
        else:
            pass
        return imgs

    def det_features(self, img1, img2,
                     img1_compute_ratio=0.3,
                     img2_comopute_ratio=0.3, out=False):
        """
        detect features between image1 and image2.
        :return matches between images,and key points for each images.
        """
        mode = self.descriptor_type
        mask1 = np.zeros(img1.shape, np.uint8)
        mask2 = np.zeros(img2.shape, np.uint8)

        keep_w1 = round(img1.shape[1] * (1 - img1_compute_ratio))
        keep_w2 = round(img2.shape[1] * img2_comopute_ratio)
        mask1[:, keep_w1:] = 1
        mask2[:, :keep_w2] = 1
        matches, keypoints1, keypoints2 = compute_matches(img1, img2, type=mode,
                                                          mask1=mask1, mask2=mask2, out=out)
        return matches, keypoints1, keypoints2

    def stitch_single(self, img1, img2,
                      img1_src=None, img2_src=None,
                      stitch_src=False, feature_ratio=0.25,
                      default_overlap_ratio=0.2, show=False,
                      ret_image=False, restore_size=False):
        matches, k1, k2 = self.det_features(img1, img2, img1_compute_ratio=feature_ratio)
        # If ret_image is True, return stitch_image,img1ori,img2ori,imgoverlap. else return img1ori,..
        stitch_info = merge_image(img1, img2, matches, k1, k2,
                                  img1_src=img1_src, img2_src=img2_src,
                                  stitch_src=stitch_src,
                                  default_overlap_ratio=default_overlap_ratio,
                                  ret_image=ret_image, restore_size=restore_size)
        if show:
            cv2.namedWindow("stitch", cv2.WINDOW_NORMAL)
            cv2.imshow("stitch", stitch_info[0])
            cv2.waitKey()
        return stitch_info

    def stitch(self, images, bboxes=None, stitch_src=False, out=None, restore_size=False):
        """
        Stitch a batch of images,where a batch contain 7 images.
        :param images: 7 images.
        :param bboxes: bboxes with respect to images. each box have 5 param.
        :param stitch_src: Bool,If stitch source image or not.
        :param out: str, out path that final stitched image saved in.
        :param restore_size: If stitch original size source images or not.
        :return:
        """
        images_h, images_w = images[0].shape[0], images[0].shape[1]
        images_norm = self.preprocess(images, is_2gray=self.is_2gray,
                                      resize_ratio=self.resize,
                                      is_norm=self.is_norm)
        imgs_overlap = []
        imgs_left = []
        imgs_right = []
        shiftys = []
        for i in range(len(images_norm)):
            if i == len(images_norm) - 1:
                continue
            # stitch 2 images and return img1ori img2ori and img overlap
            stitch_info = self.stitch_single(images_norm[i], images_norm[i + 1],
                                             img1_src=images[i], img2_src=images[i + 1], stitch_src=stitch_src,
                                             restore_size=restore_size)
            img1ori, img2ori, imgoverlap, shifty = stitch_info

            imgs_left.append(img1ori)
            imgs_right.append(img2ori)
            imgs_overlap.append(imgoverlap)
            shiftys.append(shifty)

        # keep first img and last img
        img_first = imgs_left[0]
        img_last = imgs_right[-1]

        # preserve middle images with only overlap region & independent region.
        if bboxes is not None:
            imgs_stitch = []
            shiftys = np.array(shiftys)
            bboxes_left = []
            bboxes_right= []
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
                            xmin, ymin, xmax, ymax = bbox
                            ovr_ymin = ymin - bbox_shift
                            ovr_ymax = ymax - bbox_shift
                            # bbox completely not in stitch region.Just keep it .
                            if xmax <= left_w and xmin < left_w:
                                bboxes_left.append(bbox)
                            # part of bbox in seam and other not.
                            elif xmax > left_w and xmin < left_w:
                                left_xmax = left_w
                                bboxes_left.append([xmin, ymin, left_xmax, ymax])
                                ovr_xmin = 0
                                ovr_xmax = ymax - left_w
                                bboxes_ovr.append([ovr_xmin, ovr_ymin, ovr_xmax, ovr_ymax])
                            # bbox in overlap region.
                            elif xmax > left_w and xmin >= left_w:
                                ovr_xmin = xmin - left_w
                                ovr_xmax = xmax - left_w
                                bboxes_ovr.append([ovr_xmin, ovr_ymin, ovr_xmax, ovr_ymax])

                bboxes_mid_tmp = []  # used to keep bboxes in right part.
                overlap_w = imgs_overlap[i - 1].shape[1]  # get overlap width, to compute relative coordinate of bbox.
                overlap_w2 = imgs_overlap[i].shape[1]
                right_w = imgs_right[i-1].shape[1]
                bboxes_ovr2 = []  # overlap region of right part.
                if len(bboxes_right_img) < 1:
                    pass
                else:
                    for bbox in bboxes_right_img:
                        xmin, ymin, xmax, ymax = bbox
                        ymin = ymin - bbox_shift
                        ymax = ymax - bbox_shift
                        # bbox in left overlap region.
                        if xmin < overlap_w and xmax <= overlap_w:
                            bboxes_ovr.append([xmin, ymin, xmax, ymax])
                        # part of bbox in left overlap region and other not.
                        elif xmin < overlap_w and xmax > overlap_w:
                            bboxes_ovr.append([xmin, ymin, overlap_w, ymax])
                            # bbox in middle part.
                            if xmax <= images_w - overlap_w2:
                                xmin_mid = 0
                                xmax_mid = xmax - overlap_w
                                bboxes_mid_tmp.append([xmin_mid, ymin, xmax_mid, ymax])
                            # bbox in right overlap region.
                            else:
                                xmin_mid = 0
                                xmax_mid = right_w - overlap_w2
                                bboxes_mid_tmp.append([xmin_mid, ymin, xmax_mid, ymax])
                                xmin_right = 0
                                xmax_right = xmax - right_w
                                bboxes_ovr2.append([xmin_right, ymin, xmax_right, ymax])
                        # bbox in middle region.
                        elif xmin >= overlap_w and xmax <= images_w - overlap_w2:
                            xmin_mid = xmin - overlap_w
                            xmax_mid = xmax - overlap_w
                            bboxes_mid_tmp.append([xmin_mid, ymin, xmax_mid, ymax])
                        # part of bbox in middle region,other in right overlap.
                        elif xmin >= overlap_w and xmax > images_w - overlap_w2:
                            xmin_mid = xmin - overlap_w
                            xmax_mid = right_w - overlap_w2
                            bboxes_right.append([xmin_mid, ymin, xmax_mid, ymax])
                            xmin_right = 0
                            xmax_right = xmax - right_w
                            bboxes_ovr2.append([xmin_right, ymin, xmax_right, ymax])
                        elif xmin >= images_w - overlap_w2 and xmax > images_w - overlap_w2:
                            xmin_right = xmin - right_w
                            xmax_right = xmax - right_w
                            bboxes_ovr2.append([xmin_right, ymin, xmax_right, ymax])
                if i==(len(imgs_overlap)-1):
                    bboxes_right_img=bboxes[i+1]    # get last img bbox
                    bbox_shift=shiftys.sum()
                    if len(bboxes_right_img)<1:
                        pass
                    else:
                        for bbox in bboxes_right_img:
                            xmin,ymin,xmax,ymax=bbox
                            ymin=ymin-bbox_shift
                            ymax=ymax-bbox_shift
                            if xmin<overlap_w2 and xmax < overlap_w2:
                                bboxes_ovr2.append([xmin,ymin,xmax,ymax])
                            elif xmin<overlap_w2 and xmax>overlap_w2:
                                bboxes_ovr2.append([xmin,ymin,overlap_w2,ymax])
                                xmin_right=0
                                xmax_right=xmax-overlap_w2
                                bboxes_right.append([xmin_right,ymin,xmax_right,ymax])
                            elif xmin>overlap_w2 and xmax>overlap_w2:
                                xmin_right=xmin-overlap_w2
                                xmax_right=xmax-overlap_w2
                                bboxes_right.append([xmin_right,ymin,xmax_right,ymax])
                bboxes_mid.append(bboxes_mid_tmp)
                bboxes_left_ovr.append(bboxes_ovr)
                bboxes_right_ovr.append(bboxes_ovr2)
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
            if i == len(imgs_overlap) - 1:
                imgs_stitch.append(img_last)

        stitched = cv2.hconcat(imgs_stitch)
        if out:
            # cv2.namedWindow("stitched image", cv2.WINDOW_NORMAL)
            # cv2.imshow("stitched image", stitched)
            cv2.imwrite(out, stitched)
            # cv2.waitKey()


class stitch_image(object):
    def __init__(self,
                 img_path,
                 stitchindex,
                 patch_left_ratio=1,
                 patch_right_ratio=1,
                 ):
        self.imgname = img_path.replace("\\", '/').split('/')[-1]
        self.image = cv2.imread(img_path)
        self.width = self.image.shape[1]
        self.height = self.image.shap[0]
        self.stitchindex = stitchindex
        self.patch_left_ratio = patch_left_ratio
        self.patch_right_ratio = patch_right_ratio
        self.match_patch_step = 500
        self.shift_points = None

def stitch_batch(stitcher, images, bboxes=None, out=None):
    stitcher.stitch(images, bboxes, out=out, stitch_src=True, restore_size=True)


def stitch_batches(stitcher, batches, bboxes=None, out_dir=None):
    for i, images in enumerate(batches):
        stitch_batch(stitcher, images, bboxes, out=out_dir + "src_{}.jpg".format(i))


stitch = Stitcher(descriptor_type='surf', resize=0.3)
# files = os.listdir('images')
# images = []
# for i in files:
#     images.append(cv2.imread('images/{}'.format(i)))

# 测试，读取图片的bbox信息
# path = "D:/tmp/bbox"
# bboxes = read_txt_mklist(path)

# 拼接所有图片，batches用来存取所有图片，每个batch中包含7张图
path = "/data2/yeliang/data/stitch_test"
batches = []
for j in range(7):
    images = []
    for i in range(1, 8):
        file = os.listdir(os.path.join(path, "{}".format(i)))[j]
        images.append(cv2.imread(os.path.join(os.path.join(path, "{}".format(i)), file)))
    batches.append(images)

import datetime
now = datetime.datetime.now()
stitch_batches(stitch, batches, out_dir="/data2/yeliang/data/stitch_test/results")
end=datetime.datetime.now()
print((end-now).seconds)

# image_norm = stitch.preprocess(images, is_2gray=True, resize_ratio=0.5, is_norm=True)
# stitch.stitch_single(image_norm[0], image_norm[1])
# stitch.det_features(imageg_norm[0], imageg_norm[1])


# now = datetime.datetime.now()
# stitch.stitch(images,stitch_src=True,out="./panorama_gray.jpg",restore_size=True)
# print((datetime.datetime.now() - now).seconds)

# cv2.namedWindow("origin", cv2.WINDOW_NORMAL)
# cv2.namedWindow("equalize", cv2.WINDOW_NORMAL)
# for i in range(len(images)):
#     cv2.imshow("origin", images[i])
#     cv2.imshow("equalize", image_norm[i])
#     cv2.waitKey()
