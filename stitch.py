import cv2
import numpy as np
import os
from mtcv import bgr2gray, histEqualize, draw_bboxes, nms
from mtcv import bbox_to_patch, imgs_to_patches, concat_bbox
from mtcv import read_txt_mklist
from .utils.merge import merge_image
from .utils.features import compute_matches
from .utils.misc import get_patch_shape,draw_bbox,draw_segment
import time


class Stitcher(object):
    """
    A stitching class used for image stitching.
    It consists of following modules:
    img_resize: resize img by given ratio or dsize.
    preprocess:perform preprocess orderly by resize,bgr2gray,histogram equalization.

    """
    def __init__(self, resize=0.3,
                 stitched_path=None,
                 stitched_thumbnail_path=None,
                 camera_path=None,
                 bbox_info_path=None,
                 patch_info_path=None,
                 descriptor_type='surf',
                 match_type='knn'):
        """
        :param resize: resize ratio during stitching process,smaller may more quicker.
        :param stitched_path: the output path of stitched image.
        :param stitched_thumbnail_path: the output path of thumbnail stitched image.
        :param camera_path: The path of all images,should contain Camera1-Camera7,
        :param bbox_info_path:
        :param patch_info_path:
        :param descriptor_type:
        :param match_type:
        """
        self.resize = resize
        self.camera_path = camera_path
        self.descriptor_type = descriptor_type
        self.match_type = match_type
        self.img_h = 4000
        self.img_w = 2048
        self.stitched_path = stitched_path
        self.stitched_thumbnail_path = stitched_thumbnail_path
        self.bbox_info_path = bbox_info_path
        self.patch_info_path = patch_info_path

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

    def preprocess(self, images, is_2gray=True, resize_ratio=None, is_norm=True, clipLimit=15):
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

    def start(self):
        self.Cameras, self.Camera_imgs = self.get_files_arr(self.camera_path)
        self.check_valid()

        # for i in range(137 - 28):
        for i in range(len(self.Camera_imgs[0])):
            # i = i + 28
            cam_imgs = []
            # construct a batch where contains 7 images.
            for k in range(len(self.Camera_imgs)):
                img_path = os.path.join(os.path.join(self.camera_path, self.Cameras[k]), self.Camera_imgs[k][i])
                cam_imgs.append(img_path)
            yield self.stitch(cam_imgs, stitch_src=True, out=self.stitched_path, out_thumb=self.stitched_thumbnail_path,
                              restore_size=True, segment_info=[[34482, 33375, 933, 1, 933]])

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

    def stitch(self, images_path,
               bboxes=None,
               stitch_src=True,
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
        images_norm = self.preprocess(images,resize_ratio=self.resize)

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
        coordinates, imgs_left_shape, imgs_overlap_shape, imgs_right_shape=get_patch_shape(
            imgs_stitch,imgs_left,imgs_overlap,imgs_right)

        stitched = cv2.hconcat(imgs_stitch)
        if bboxes is not None:
            bboxes_all =draw_bbox(bboxes,shiftys,imgs_left_shape,imgs_overlap_shape,
                                       imgs_right_shape,coordinates,images_w)
            stitched = draw_bboxes(stitched, bboxes_all)
        segment_ids=[]
        if segment_info:
            stitched,segment_ids = draw_segment(stitched, stitched_name, *segment_info, concat_width=500, x_width=20)

        if out:
            out_file = os.path.join(out, stitched_name)
            cv2.imwrite(out_file, stitched)
            if out_thumb:
                cv2.imwrite(out_thumb, stitched, [int(cv2.IMWRITE_JPEG_QUALITY), save_quality])
        return out, out_thumb, coordinates,segment_ids



# test

stitcher = Stitcher(resize=0.3, camera_path='/data2/yeliang/data/tunnel_camera/20190703_line17_camera',
                    stitched_path='/data2/yeliang/data/tunnel_camera/line17_stitch')
generator=stitcher.start()
i=1
num=0
while True:
    try:
        # start=datetime.datetime.now()
        start=time.time()
        info=generator.__next__()
        # end=datetime.datetime.now()
        end=time.time()
        single_cost=round(end-start,3)
        num+=single_cost
        print("time cost in stitching {} image:".format(i),single_cost)
        print("total cost:",num)
        i+=1
    except StopIteration:
        break

# for i in range(109):
#     result = next(generator)

def stitch_batch(stitcher, images, bboxes=None, out=None):
    stitcher.stitch(images, bboxes, out=out, stitch_src=True, restore_size=True)


def stitch_batches(stitcher, batches, bboxes=None, out_dir=None):
    for i, images in enumerate(batches):
        stitch_batch(stitcher, images, bboxes[i], out=out_dir)


# stitch = Stitcher(descriptor_type='surf', resize=0.3)
# files = os.listdir('images')
# images = []
# for i in files:
#     images.append(cv2.imread('images/{}'.format(i)))

# 测试，读取图片的bbox信息
# bbox_path = "/data2/yeliang/data/stitch_test/bbox"
# bboxes = read_txt_mklist(bbox_path)

# 拼接所有图片，batches用来存取所有图片，每个batch中包含7张图
# path = "/data2/yeliang/data/stitch_test"
# batches = []
# for j in range(7):
#     images = []
#     for i in range(1, 8):
#         file = os.listdir(os.path.join(path, "Camera{}".format(i)))
#         file.sort()
#         file = file[j]
#         images.append(os.path.join(os.path.join(path, "Camera{}".format(i)), file))
#         # images.append(cv2.imread(os.path.join(os.path.join(path, "Camera{}".format(i)), file)))
#     batches.append(images)



# now = datetime.datetime.now()
# stitch_batches(stitch, batches, bboxes, out_dir="/data2/yeliang/data/stitch_test/results/")
# end = datetime.datetime.now()
# print((end - now).seconds)

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
