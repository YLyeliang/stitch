# import the necessary packages
import os
import numpy as np
import argparse
import cv2

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
image_paths="images/"
imagePaths = os.listdir(image_paths)
images = []

# loop over the image paths, load each one, and add them to our
# images to stitch list
for i,imagePath in enumerate(imagePaths):
    image = cv2.imread(os.path.join('images',imagePath))
    if i ==2 or i ==3:
        images.append(image)

# initialize OpenCV's image stitcher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher()
(status, stitched) = stitcher.stitch(images)

# if the status is '0', then OpenCV successfully performed image
# stitching
if status == 0:
    # write the output stitched image to disk
    cv2.imwrite('./results.jpg', stitched)

    # display the output stitched image to our screen
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)

# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
    print("[INFO] image stitching failed ({})".format(status))

