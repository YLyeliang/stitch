import cv2
import numpy as np
import os
from mtcv import bgr2gray, histEqualize,draw_bboxes,nms
from mtcv import bbox_to_patch,imgs_to_patches,concat_bbox
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
                 descriptor_type='surf',
                 match_type='knn',
                 show=True):
        self.resize = resize
        self.descriptor_type = descriptor_type
        self.match_type = match_type
        self.is_norm = is_norm
        self.is_2gray = is_2gray
        self.show = show
        self.img_h=4000
        self.img_w=2048

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

    def concat_segment(self,width=500):
        """
        Add segment information on the right side of image.
        :return:
        """
        r = np.zeros((width,self.img_h),dtype=np.uint8)
        g = np.zeros((width, self.img_h), dtype=np.uint8)
        b = np.zeros((width, self.img_h), dtype=np.uint8)
        rgb=np.stack((b,g,r))


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

    def stitch(self, images,
               bboxes=None,
               stitch_src=False,
               out=None,
               restore_size=False,
               save_quality=100):
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

        # preserve middle images with only overlap region & independent region.
        shiftys = np.array(shiftys)
        imgs_stitch=imgs_to_patches(imgs_left,imgs_overlap,imgs_right,shiftys)
        patches_width=[patch.shape[1] for patch in imgs_stitch]
        patches_width=np.array(patches_width)
        coordinates=[patches_width[:i+1].sum() for i in range(len(patches_width))]

        stitched = cv2.hconcat(imgs_stitch)
        if bboxes is not None:
            bboxes_all=bbox_to_patch(bboxes,shiftys,imgs_left,imgs_overlap,imgs_right,
                                     imgs_stitch,images_w)
            bboxes_all=concat_bbox(bboxes_all)
            self.bboxes_all,inds=nms(bboxes_all)
            self.stitched = draw_bboxes(stitched, self.bboxes_all)

        if out:
            # cv2.namedWindow("stitched image", cv2.WINDOW_NORMAL)
            # cv2.imshow("stitched image", stitched)
            cv2.imwrite(out, self.stitched,[int(cv2.IMWRITE_JPEG_QUALITY),save_quality])
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
        stitch_batch(stitcher, images, bboxes[i], out=out_dir + "src_{}.jpg".format(i))

stitch = Stitcher(descriptor_type='surf', resize=0.3)
# files = os.listdir('images')
# images = []
# for i in files:
#     images.append(cv2.imread('images/{}'.format(i)))

# 测试，读取图片的bbox信息
bbox_path = "/data2/yeliang/data/stitch_test/bbox"
bboxes = read_txt_mklist(bbox_path)

# 拼接所有图片，batches用来存取所有图片，每个batch中包含7张图
path = "/data2/yeliang/data/stitch_test"
batches = []
for j in range(7):
    images = []
    for i in range(1, 8):
        file = os.listdir(os.path.join(path, "{}".format(i)))
        file.sort()
        file = file[j]
        images.append(cv2.imread(os.path.join(os.path.join(path, "{}".format(i)), file)))
    batches.append(images)

import datetime

now = datetime.datetime.now()
stitch_batches(stitch, batches, bboxes, out_dir="/data2/yeliang/data/stitch_test/results/")
end = datetime.datetime.now()
print((end - now).seconds)

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
