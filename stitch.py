import cv2
import numpy as np
import os
from mtcv import bgr2gray, histEqualize, draw_bboxes, nms
from mtcv import bbox_to_patch, imgs_to_patches, concat_bbox
from mtcv import read_txt_mklist
from utils.merge import merge_image
from utils.features import compute_matches
import re
import math


class Stitcher(object):
    """
    A stitching class used for image stitching.
    It consists of following modules:
    img_resize: resize img by given ratio or dsize.
    preprocess:perform preprocess orderly by resize,bgr2gray,histogram equalization.

    """

    def __init__(self, resize=0.5,
                 stitched_path=None,
                 stitched_thumbnail_path=None,
                 camera_path=None,
                 is_norm=True,
                 is_2gray=True,
                 bbox_info_path=None,
                 patch_info_path=None,
                 descriptor_type='surf',
                 match_type='knn',
                 show=True):
        self.resize = resize
        self.camera_path = camera_path
        self.descriptor_type = descriptor_type
        self.match_type = match_type
        self.is_norm = is_norm
        self.is_2gray = is_2gray
        self.show = show
        self.img_h = 4000
        self.img_w = 2048
        self.stitched_path = stitched_path
        self.stitched_thumbnail_path = stitched_thumbnail_path
        self.bbox_info_path = bbox_info_path
        self.patch_info_path = patch_info_path
        self.Cameras, self.Camera_imgs = self.get_files_arr(camera_path)

    def get_files_arr(self, path):
        """By given path,automatically reading director of name "Camera1" to "Camera7"
        :arg
        :param path:should containing 7 directory named from 'Camera1' to 'Camera7'
        """
        Cameras = ['Camera1', 'Camera2', 'Camera3', 'Camera4', 'Camera5', 'Camera6', 'Camera7']
        Camera_path = os.listdir(path)
        for i in range(len(Cameras)):
            tmp = 0
            for j in Camera_path:
                if Cameras[i] == j: tmp += 1
            if tmp == 0:
                raise FileNotFoundError("{} not found".format(Cameras[i]))
        Camera_imgs = []
        for Camera in Cameras:
            imgs = os.listdir(os.path.join(path, Camera))
            imgs.sort()
            Camera_imgs.append(imgs)
        return Cameras, Camera_imgs

    def read_img(self, img_path):
        """
        Read a batch of images containing 7 images,and return a list of image arrays,
        and stitched_name.
        :param img_path: (list) containing 7 images abs path.
        :return:
        """
        if isinstance(img_path, list):
            image_name = img_path[0].replace('\\', '/').split('/')[-1].split('_')
            images = []
            kmeters = []
            meters = []
            for i in img_path:
                # decode km and m in image name.
                i = i.replace("\\", '/')
                file_name = i.split('/')[-1]
                miles = file_name.split('_')[5]
                km, m = miles.split('+')
                km = int(km[:-2])
                m = float(m[:-1])
                kmeters.append(km)
                meters.append(m)
                image = cv2.imread(i)
                images.append(image)

            kmeters = np.array(kmeters)
            meters = np.array(meters)
            if kmeters.mean() == kmeters[0]:
                meter = meters.mean()
            else:
                meter = meters[3]
            stitched_name = '_'.join(image_name[1:5]) + '_' + str(kmeters[0]) + 'km+{:.1f}m_'.format(meter) + ''.join(
                image_name[6:])
        elif isinstance(img_path, str):
            images = cv2.imread(img_path)
            stitched_name = None
        else:
            raise TypeError("image path should be str or list of str containing abs path")
        return images, stitched_name

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

    def preprocess(self, images, is_2gray=True, resize_ratio=None, is_norm=False, clipLimit=15):
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
                    imgs[i] = histEqualize(img, 'clahe', clipLimit)
        # else what we read is string containing img paths.
        else:
            pass
        return imgs

    def concat_segment(self, stitched_img, stitched_name, mile_start=0, mile_end=0, segment_num=0, segment_start=1,
                       segment_end=0, concat_width=500, x_width=20):
        """
        Add segment information on the right side of image.
        mile start and end should specified,thus which section image are loacted at
        can be calculated.
        :return:
        """
        # transfer image miles str to numbers
        miles = stitched_name.split('_')[4]
        km, m = miles.split('+')
        km = int(km[:-2])
        m = float(m[:-1])
        mile = km * 1000 + m
        distance = abs(mile_start - mile_end)
        if mile_start < mile_end:
            mile_to_start = mile - mile_start
            mile_to_end = mile_end - mile
        else:
            mile_to_start = mile_start - mile
            mile_to_end = mile - mile_end

        if mile_to_start > -10 and mile_to_end > 0:
            img_h = stitched_img.shape[0]
            segment_width = abs(mile_end - mile_start) / segment_num  # segment width, e.g 1.2m
            seg_num_img = round(10 / segment_width)  # number of segment on single image
            seg_pix_integral = round(img_h / seg_num_img)

            # calculate index of segment,
            # get y coordinates of all segment
            order = segment_start < segment_end
            y_cord = [y * seg_pix_integral for y in range(seg_num_img)]
            seg_num_div = int(mile_to_start // 10)  # 计算距里程起点经过的管片数量，单张图片管片数*这个数
            seg_num_mod = round(mile_to_start % 10)
            seg_num_pass = seg_num_div * seg_num_img + round(seg_num_mod / segment_width)
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
                cv2.putText(bgr, "#{}".format(segment_id[i]), (x_width, 300+y_cord[i]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 5,
                            color=(0, 0, 255), thickness=3)
            stitched_img = cv2.hconcat([stitched_img,bgr])
        return stitched_img

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

    def check_valid(self):
        Camera1, Camera2, Camera3, Camera4, Camera5, Camera6, Camera7 = self.Camera_imgs
        assert len(Camera1) == len(Camera2) == len(Camera3) == len(Camera4) == len(Camera5) == len(Camera6) == len(
            Camera7)

        # check whether 7 images are on the same line.
        for i in range(len(Camera1)):
            cmp_names = [self.Camera_imgs[j][i].split('_')[4] for j in range(len(self.Camera_imgs))]
            for i_ in range(len(cmp_names) - 1):
                if not cmp_names[i_] == cmp_names[i_ + 1]:
                    raise ValueError(
                        "Images to be stitched are not the same line.{} & {}".format(self.Camera_imgs[i_][i],
                                                                                     self.Camera_imgs[i_ + 1][i]))

    def start(self):
        Camera1, Camera2, Camera3, Camera4, Camera5, Camera6, Camera7 = self.Camera_imgs

        for i in range(137-29):
        # for i in range(len(self.Camera_imgs[0])):
            i = i + 28
            cam_imgs = []
            # construct a batch where contains 7 images.
            for k in range(len(self.Camera_imgs)):
                img_path = os.path.join(os.path.join(self.camera_path, self.Cameras[k]), self.Camera_imgs[k][i])
                cam_imgs.append(img_path)
                # cam_img=cv2.imread(img_path)
                # cam_imgs.append(cam_img)
            self.stitch(cam_imgs, stitch_src=True, out=self.stitched_path, out_thumb=self.stitched_thumbnail_path,
                        restore_size=True, segment_info=[34482, 33370, 933, 1, 933])

    def stitch(self, images_path,
               bboxes=None,
               stitch_src=False,
               out=None,
               out_thumb=None,
               restore_size=True,
               segment_info=None,
               save_quality=50):
        """
        Stitch a batch of images,where a batch contain 7 images.
        :param images: 7 images.
        :param bboxes: bboxes with respect to images. each box have 5 param.
        :param stitch_src: Bool,Whether stitch source image or not.
        :param out: str, out path that final stitched image saved in.
        :param restore_size: If stitch original size source images or not.
        :param segment_info: (list) [mile_start,mile_end,segment_num,segment_start,segment_end,]
        :return:
        """
        # read image and get stitched name.
        images, stitched_name = self.read_img(images_path)
        images_h, images_w = images[0].shape[0], images[0].shape[1]

        # preprocess images to better stitch.
        images_norm = self.preprocess(images, is_2gray=self.is_2gray,
                                      resize_ratio=self.resize,
                                      is_norm=self.is_norm)

        # get stitched patches information,which contains 6 left patch images,
        # 6 right patch images, 6 middle patches images, and vertical shifts.
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
        imgs_stitch = imgs_to_patches(imgs_left, imgs_overlap, imgs_right, shiftys)
        patches_width = [patch.shape[1] for patch in imgs_stitch]
        patches_width = np.array(patches_width)  # patches num: 2*i-1
        coordinates = [patches_width[:i + 1].sum() for i in range(len(patches_width))]

        stitched = cv2.hconcat(imgs_stitch)
        if bboxes is not None:
            bboxes_all = bbox_to_patch(bboxes, shiftys, imgs_left, imgs_overlap, imgs_right,
                                       imgs_stitch, images_w)
            bboxes_all = concat_bbox(bboxes_all)
            bboxes_all, inds = nms(bboxes_all)
            stitched = draw_bboxes(stitched, bboxes_all)
        if segment_info:
            stitched = self.concat_segment(stitched, stitched_name, *segment_info, concat_width=500, x_width=20)

        if out:
            # cv2.namedWindow("stitched image", cv2.WINDOW_NORMAL)
            # cv2.imshow("stitched image", stitched)
            out_file=os.path.join(out,stitched_name)
            cv2.imwrite(out_file, stitched)
            if out_thumb:
                cv2.imwrite(out_thumb, stitched, [int(cv2.IMWRITE_JPEG_QUALITY), save_quality])
            # cv2.waitKey()
        return out, out_thumb, coordinates


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


# test
stitcher = Stitcher(resize=0.3, camera_path='/data2/yeliang/data/tunnel_camera/20190703_line17_camera',
                    stitched_path='/data2/yeliang/data/tunnel_camera/line17_stitch')
stitcher.start()


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
        file = os.listdir(os.path.join(path, "Camera{}".format(i)))
        file.sort()
        file = file[j]
        images.append(cv2.imread(os.path.join(os.path.join(path, "Camera{}".format(i)), file)))
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
