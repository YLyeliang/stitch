import numpy as np
import cv2

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
        img2crop = img2[:img2h + shifty, :]  # 裁剪右图，使其与左图对齐
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
        # optimization version B
        ind=np.arange(imgow)
        w=np.empty(imgow)
        w.fill(imgow)
        alpha=(w-ind)/w
        beta=np.ones(imgow)-alpha
        for i in range(imgoverlap.shape[2]):
            imgoverlap[:,:,i]=img1overlap[:,:,i]*alpha+img2overlap[:,:,i]*beta

        # optimization version A
        # imgoverlap[:,:,:]=img1overlap*alpha+img2overlap*beta
        # for j in range(imgow):
        #     alpha = float(imgow - j) / imgow
        #     imgoverlap[:, j, :] = img1overlap[:, j, :] * alpha + img2overlap[:, j, :] * (1.0 - alpha)

        # original version
        # for i in range(imgoh):
        #     for j in range(imgow):
        #         if img2overlap[i, j, 0] == 0 and img2overlap[i, j, 1] == 0 and img2overlap[i, j, 2] == 0:
        #             alpha = 1.0
        #         else:
        #             alpha = float(imgow - j) / imgow
        #         imgoverlap[i, j, :] = img1overlap[i, j, :] * alpha + img2overlap[i, j, :] * (1.0 - alpha)

    else:  # 灰度图像拼接
        for j in range(imgow):
            alpha = float(imgow - j) / imgow
            imgoverlap[:, j] = int(img1overlap[:, j] * alpha + img2overlap[:, j] * (1.0 - alpha))

        # for i in range(imgoh):
        #     for j in range(imgow):
        #         if img2overlap[i, j] == 0:
        #             alpha = 1.0
        #         else:
        #             alpha = float(imgow - j) / imgow
        #         imgoverlap[i, j] = int(img1overlap[i, j] * alpha + img2overlap[i, j] * (1.0 - alpha))
    final = cv2.hconcat([img1ori, imgoverlap, img2ori])
    return final, img1ori, img2ori, imgoverlap, shifty