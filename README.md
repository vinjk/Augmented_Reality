# Augmented Reality using Python and OpenCV

This is an OpenCV project to
1. Track a surface in a video
2. Get the camera position
3. Render a 3D object on to the video

The inspiration for this work was from a paper from Disney Research Magnenat et al (2015), 'Live Texturing of Augmented Reality Characters from Colored Drawings'. You can find more about the work here: https://www.disneyresearch.com/publication/live-texturing-of-augmented-reality-characters/

In this work I have used OpenCV library for implementing the project. Some of the key techniques I used are:
1. Camera calibration (chessboard pattern)
2. Extraction of line drawing from coloured drawings
3. BRISK feature detector (cv2.BRISK_create, brisk.detectAndCompute)
4. FLANN feature matching (cv2.FlannBasedMatcher, flann.knnMatch)
5. Optical flow tracking (cv2.calcOpticalFlowPyrLK)
6. Compute Homography (cv2.findHomography)
7. Camera pose estimation (cv2.solvePnPRansac)
8. Simple plotting of cube using cv2.line function
