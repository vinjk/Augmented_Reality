import cv2
import numpy as np
import process_func as pf


dataset_path = '../data/dataset1/'
param_path = '../data/param/'
template_path = '../data/templates/'

template_filename = template_path + 'template.jpg'
video_filename = dataset_path + 'colour_vid.mp4'
camera_params_filename = param_path + 'cam_intrinsic_distort.npz'
output_filename = dataset_path + 'output.mp4'

# 3D model points in world coordinates
# Corners of the colouring page
pg_points = np.array([
    (93.0, 135.0, 0.0),  # 1
    (93.0, -135.0, 0.0),  # 2
    (-93.0, -135.0, 0.0),  # 3
    (-93.0, 135.0, 0.0)  # 4
])

# Load camera matrix and distortion coefficient from camera calibration
# The phone camera was calibrated usign checkerboard pattern
cam_params = np.load(camera_params_filename)
camera_matrix = cam_params['camera_matrix']
dist_coefs = cam_params['dist_coefs']

# Define codec to write output video
# Define the codec and create VideoWriter object
#for fourcc, the ASCII is directly provided as the cv2.VideoWriter_fourcc() doesn't work
out = cv2.VideoWriter(output_filename, 0x00000021, 25.0, (960,540))

# The colouring page- original drawing
# In the case of this implementation, an already coloured image was used. The image_proc function is used to remove all
# the colours and extract just the line drawing.
img_org = cv2.imread(template_filename)

# Resize image for image size reduction
scale_factor = 0.25
# img1 = pf.image_proc(cv2.resize(img_org,None,fx=scale_factor,fy=scale_factor),scale_factor)
img1 = pf.image_proc(cv2.resize(img_org,(540,960)),scale_factor)

# Load the video and read out the first frame and process it to extract the line drawing
cap = cv2.VideoCapture(video_filename)
_,img_fframe = cap.read()
img_fframe_resize = cv2.resize(img_fframe, None, fx=0.5, fy=0.5)
img2_fframe = pf.image_proc(img_fframe_resize, 0.5)

# STAGE 1, where the features from the first video frame and the template image is identified and matched
# The features identified in this stage is used to track the colouring page in the subsequent video frames

# Feature identification and matching
dst_pts, dst = pf.brisk_flann(img1, img2_fframe)

# draw frame boundary and display in video
img_marked = pf.draw_frame(img_fframe_resize, dst)
cv2.imshow('Video',img_marked)

# STAGE-2 where the features identified will be used to track (Lucas-Kanade optical flow) the colouring in video
# In this stage, we estimate homography and camera pose and use it to render a cube in the video frame in real-time

# Copy feature points and image frame. The feature points from brisk-flann will be used in optical flow tracking
src_pts = np.copy(dst_pts)
img2_old = np.copy(img2_fframe)

# Setup parameters for optical tracking in video
# Parameters for Shi-Tomasi corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
#
# src_pts = cv2.goodFeaturesToTrack(img2_fframe, mask = None, **feature_params)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Read and process frames from video
while True:
    # Write the frame into the file 'output.mp4'
    out.write(img_marked)

    # Read frame
    ret, img_scn = cap.read()

    if ret:
        # Resize frame to smaller size
        img_scn_resize = cv2.resize(img_scn, None, fx=0.5, fy=0.5)

        # Remove all colours and make frame close to original template as possible
        img2 = pf.image_proc(img_scn_resize, 0.5)

        # Calculate optical flow
        dst_pts, st, err = cv2.calcOpticalFlowPyrLK(img2_old, img2, src_pts, None, **lk_params)

        # Select good points
        good_new = dst_pts[st == 1]
        good_old = src_pts[st == 1]

        # Compute Homography
        M = pf.computeHomography(good_old, good_new)

        #Transform frame edge based on new homography
        dst = cv2.perspectiveTransform(dst, M)

        # draw frame boundary and display in video
        img_marked = pf.draw_frame(img_scn_resize, dst)

        # Copy feature points and frame for processing of next frame
        src_pts = np.copy(good_new).reshape(-1,1,2)
        img2_old = np.copy(img2)

        # Estimate the camera pose from frame corner points in world coordinates and image frame
        # THe rotation vectors and translation vectors are obtained
        ret, rvecs, tvecs, inlier_pt = cv2.solvePnPRansac(pg_points, dst, camera_matrix, dist_coefs)

        # Render cube in the video
        # Project cube corners in world coordinates to image frame
        # Two faces (top and bottom are shown. They are connected by red lines.
        img_marked = pf.plot_cube(img_marked, rvecs, tvecs, camera_matrix, dist_coefs)

        # Display in video
        cv2.imshow('Video', img_marked)

        # Press 'q' on keyboard to exit program
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        print 'End of video'
        break

# Close all windows and release video capture object
cv2.destroyAllWindows()
cap.release()