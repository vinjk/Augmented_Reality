# Augmented Reality using OpenCV library

This is an OpenCV project to
1. Process the coloured drawing and extract the original line drawing
2. Track a surface in a video
3. Get the camera position
4. Render a 3D object on to the video

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

I have mainly relied on opencv documentations to understand how to use its functions and the above mentioned paper to implement this project. Special thanks to the wonderful resources on www.learnopencv.com run by [@spmallick](https://github.com/spmallick) and also the [blog article](https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/) by [@juangallostra](https://github.com/juangallostra) (I see he has published Part 2 as well!)
