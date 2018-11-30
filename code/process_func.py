# Implementation of all required functions for the project

import cv2
import numpy as np

# Process input image to match the original line drawing
def image_proc(img,scale_factor):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    #Luminance channel of HSV image
    lum = img_hsv[:,:,2]

    #Adaptive thresholding
    lum_thresh = cv2.adaptiveThreshold(lum,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,15)

    #Remove all small connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(lum_thresh, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 90*scale_factor

    lum_clean = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            lum_clean[output == i + 1] = 255

    # use mask to remove all neat outline of original image
    lum_seg = np.copy(lum)
    lum_seg[lum_clean!=0] = 0
    lum_seg[lum_clean==0] = 255

    # Gaussian smoothing of the lines
    lum_seg = cv2.GaussianBlur(lum_seg,(3,3),1)

    return lum_seg

# Compute the homography
def computeHomography(src_pts, dst_pts):
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

    return M

# Draw lines for frame boundary
def draw_frame(img,dst):
    img = cv2.polylines(img, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

    return img

# Feauture identification and matching using BRISK detector and FLANN feature matching
def brisk_flann(img1, img2):
    # Initiate BRISK detector
    brisk = cv2.BRISK_create()

    kp1, des1 = brisk.detectAndCompute(img1, None)
    kp2, des2 = brisk.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_LSH = 1
    index_params = dict(algorithm=6,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=1)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 50
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = computeHomography(src_pts, dst_pts)

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

    else:
        print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)

    return dst_pts, dst

# Plot cube in current frame of video
def plot_cube(img_marked, rvecs, tvecs, camera_matrix, dist_coefs):
    # Cube corner points in world coordinates
    axis8 = np.float32([[0, 0, 0], [12, 0, 0], [12, 12, 0], [0, 12, 0], [0, 0, -12], [12, 0, -12], [12, 12, -12],
                        [0, 12, -12]]).reshape(-1, 3)

    # Project corner points of the cube in image frame
    imgpts, jac = cv2.projectPoints(axis8, rvecs, tvecs, camera_matrix, dist_coefs)

    # Render cube in the video
    # Two faces (top and bottom are shown. They are connected by red lines.
    imgpts = np.int32(imgpts).reshape(-1, 2)
    face1 = imgpts[:4]
    face2 = np.array([imgpts[0], imgpts[1], imgpts[5], imgpts[4]])
    face3 = np.array([imgpts[2], imgpts[3], imgpts[7], imgpts[6]])
    face4 = imgpts[4:]

    # Bottom face
    img = cv2.drawContours(img_marked, [face1], -1, (255, 0, 0), -3)

    # Draw lines connected the two faces
    img = cv2.line(img_marked, tuple(imgpts[0]), tuple(imgpts[4]), (0, 0, 255), 2)
    img = cv2.line(img_marked, tuple(imgpts[1]), tuple(imgpts[5]), (0, 0, 255), 2)
    img = cv2.line(img_marked, tuple(imgpts[2]), tuple(imgpts[6]), (0, 0, 255), 2)
    img = cv2.line(img_marked, tuple(imgpts[3]), tuple(imgpts[7]), (0, 0, 255), 2)

    # Top face
    img = cv2.drawContours(img_marked, [face4], -1, (0, 255, 0), -3)

    return img_marked