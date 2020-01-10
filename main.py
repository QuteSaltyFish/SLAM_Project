import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def calibrate_camera():
    # Chechboard dimensions
    cbrow = 13
    cbcol = 11

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('516030910254-陈烨/相机内参标定图片/*.JPG')
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


if __name__ == "__main__":
    img = cv2.imread('516030910254-陈烨/3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (680, 480))
    # img.resi
    mtx, dist = calibrate_camera()
    dst = undistortion(img, mtx, dist)
    cv2.imwrite('result.jpg', dst)
