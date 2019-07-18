import cv2
import math
import numpy as np

feature_type = ['sift', 'surf']


def compute_matches(img1,
                    img2,
                    type='sift',
                    mask1=None,
                    mask2=None,
                    ratio_test_threshold=0.95,
                    out=False):
    """
    SIFT descriptor

    """
    i1 = img1.copy()
    i2 = img2.copy()
    # Initiate SIFT detector
    if type == 'sift':
        feat = cv2.xfeatures2d.SIFT_create()
    elif type == 'surf':
        feat = cv2.xfeatures2d.SURF_create()
    elif type not in feature_type:
        raise NotImplementedError("{} not supported yet.".format(type))

    # Detects keypoints and computes the descriptors
    k1, des1 = feat.detectAndCompute(i1, mask1)
    k2, des2 = feat.detectAndCompute(i2, mask2)
    if des1 is None or des2 is None:
        return [], des1, des2
    # BFMatcher with default params
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf = cv2.BFMatcher()
    # Finds the k best matches for each descriptor from a query set.
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    final_matches = []
    img1h = img1.shape[0]
    img1w = img1.shape[1]
    img2h = img2.shape[0]
    img2w = img2.shape[1]
    for u, v in matches:
        if u.distance < ratio_test_threshold * v.distance:
            final_matches.append([u])
    final_matches_filter = []
    anglelst = []
    for matchinfo in final_matches:
        pt1 = k1[matchinfo[0].queryIdx].pt  # 第一张图特征点坐标
        pt2 = k2[matchinfo[0].trainIdx].pt  # 第二张图特征点坐标

        dy = pt2[1] - pt1[1]
        dx = pt2[0] + img1w - pt1[0]
        angle = math.atan(float(dy) / dx)

        if math.fabs(pt1[1] - pt2[1]) < img1h / 50:
            final_matches_filter.append(matchinfo)
            anglelst.append(angle)

    # cv2.drawMatchesKnn expects list of lists as matches.
    img_out = cv2.drawMatchesKnn(img1, k1, img2, k2, final_matches_filter, None, flags=2)  # final_matches

    if out:
        # cv2.imwrite(out, img_out)
        cv2.namedWindow("SIFT Matches", cv2.WINDOW_NORMAL)
        cv2.imshow("SIFT Matches", img_out)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return final_matches_filter, k1, k2
