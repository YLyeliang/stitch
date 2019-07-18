
import cv2
import numpy as np


# version 1:
def stitch(self, images ,show=True):
    images_norm = self.preprocess(images ,is_2gray=self.is_2gray,
                                  resize_ratio=self.resize,
                                  is_norm=self.is_norm)
    imgs_overlap = []
    imgs_left = []
    imgs_right = []
    shiftys=[]
    for i in range(len(images_norm)):
        if i == len(images_norm) - 1:
            continue
        # stitch 2 images and return img1ori img2ori and img overlap
        stitch_info = self.stitch_single(images_norm[i], images_norm[i + 1])
        img1ori ,img2ori ,imgoverlap,shifty =stitch_info

        imgs_left.append(img1ori)
        imgs_right.append(img2ori)
        imgs_overlap.append(imgoverlap)
        shiftys.append(shifty)

    # keep first img and last img
    img_first =imgs_left[0]
    img_last =imgs_right[-1]

    # preserve middle images with only overlap region & independent region.
    imgs_stitch =[]
    for i in range(1 ,len(imgs_overlap)):
        if i ==1:
            imgs_stitch.append(img_first)
            imgs_stitch.append(imgs_overlap[0])
        # all we need are current left img, and previous width of overlap.
        # then get left[:,width:],are middle img preserved to stitch.
        # since there may be shift in different images,so we need to
        # refine them.
        img_overlap=imgs_overlap[i]
        ovr_cur_w,ovr_cur_h =img_overlap.shape[1],img_overlap.shape[0]
        img_right =imgs_right[ i -1]

        img_mid =img_right[: ,:img_right.shape[1] -ovr_cur_w]

        imgs_stitch.append(img_mid)
        imgs_stitch.append(img_overlap)
        if i == len(imgs_overlap) - 1:
            imgs_stitch.append(img_last)
    stitched =cv2.hconcat(imgs_stitch)
    if show:
        cv2.namedWindow("stitched image" ,cv2.WINDOW_NORMAL)
        cv2.imshow("stitched image" ,stitched)
        cv2.imwrite("test.jpg" ,stitched)
        cv2.waitKey()


# version 2:  计算6张拼接图，对6张拼接图再次两两拼接成3张，最后做融合
# def stitch(self, images,show=True):
#     images_norm = self.preprocess(images,is_2gray=self.is_2gray,
#                                   resize_ratio=self.resize,
#                                   is_norm=self.is_norm)
#     imgs_stitch=[]
#     for i in range(len(images_norm)):
#         if i == len(images_norm) - 1:
#             continue
#         # stitch 2 images and return img1ori img2ori and img overlap
#         stitch_info = self.stitch_single(images_norm[i], images_norm[i + 1],ret_image=True)
#         imgs_stitch.append(stitch_info[0])
#
#     imgs_overlap = []
#     imgs_left = []
#     imgs_right = []
#     for i in range(math.floor(len(imgs_stitch)/2)):
#         stitch_info = self.stitch_single(imgs_stitch[i*2],imgs_stitch[2*i+1],feature_ratio=0.6)
#         img1ori, img2ori, imgoverlap = stitch_info
#         imgs_left.append(img1ori)
#         imgs_right.append(img2ori)
#         imgs_overlap.append(imgoverlap)
#
#     # keep first img and last img
#     img_first = imgs_left[0]
#     img_last = imgs_right[-1]
#     # preserve middle images with only overlap region & independent region.
#     imgs_stitch=[]
#     for i in range(1,len(imgs_overlap)):
#         if i ==1:
#             imgs_stitch.append(img_first)
#             imgs_stitch.append(imgs_overlap[0])
#         # all we need are current left img, and previous width of overlap.
#         # then get left[:,width:],are middle img preserved to stitch.
#         ovr_cur_w=imgs_overlap[i].shape[1]
#         img_right=imgs_right[i-1]
#         img_mid=img_right[:,:img_right.shape[1]-ovr_cur_w]
#         imgs_stitch.append(img_mid)
#         imgs_stitch.append(imgs_overlap[i])
#         if i == len(imgs_overlap) - 1:
#             imgs_stitch.append(img_last)
#     stitched=cv2.hconcat(imgs_stitch)
#     if show:
#         cv2.namedWindow("stitched image",cv2.WINDOW_NORMAL)
#         cv2.imshow("stitched image",stitched)
#         # cv2.imwrite("test.jpg",stitched)
#         cv2.waitKey()