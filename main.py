import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from resources.ransec import match


def calibrate_camera():
    # Chechboard dimensions
    cbrow = 13
    cbcol = 11

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('legecy/calipic/*.JPG')
    print(images)
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)
    return mtx, dist


def undistortion(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    if roi != (0, 0, 0, 0):
        dst = dst[y:y + h, x:x + w]

    return dst


def drawMatches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1+cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2, cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1), int(y1)),
                 (int(x2)+cols1, int(y2)), (255, 0, 0), 1)

    # Show the image
    cv2.imwrite('match.jpg', out)

    # Also return the image if you'd like a copy
    return out


if __name__ == "__main__":
    img1 = cv2.imread('legecy/1.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (680, 480))

    img2 = cv2.imread('legecy/2.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, (680, 480))

    mtx, dist = calibrate_camera()
    dst1 = undistortion(img1, mtx, dist)
    dst2 = undistortion(img2, mtx, dist)
    cv2.imwrite('result.jpg', dst1)

    match(img1, img2, mtx)
    # img = cv2.imread('img.jpg')

    # # Create SURF object. You can specify params here or later.
    # # Here I set Hessian Threshold to 400
    # # surf = cv2.Feature2D.descriptorType('SURF')
    # surf = cv2.xfeatures2d.SURF_create(400)
    # # tmp = cv2.SURF()
    # # Find keypoints and descriptors directly
    # kp1, des1 = surf.detectAndCompute(img1, None)
    # kp2, des2 = surf.detectAndCompute(img2, None)

    # bf = cv2.BFMatcher()
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # img3 = drawMatches(img1, kp1, img2, kp2, matches[:100])
    # # plt.imsave(img3, 'result2.jpg')
    # # print(kp, des)
