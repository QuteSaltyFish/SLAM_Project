import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D


def cameraPoseToExtrinsics(ori, loc):
    M_rot = np.linalg.inv(ori).squeeze()
    M_trans = -np.matmul(ori, loc)
    return M_rot, M_trans


def cameraMatrix(mtx, R, t):
    tmp = t.reshape(3, 1)
    tmp = np.column_stack([R, tmp])
    return np.matmul(mtx, tmp)


def match(img1, img2, mtx):

    # img1 = cv2.imread("legecy/1.jpg", 0)
    # img2 = cv2.imread("legecy/2.jpg", 0)

    # detector = cv2.ORB_create()
    detector = cv2.xfeatures2d_SURF.create(400)

    flann_params = dict(algorithm=6,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2

    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # matcher = cv2.FlannBasedMatcher(flann_params)
    FLANN_INDEX_KDTREE = 0

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    # 使用FlannBasedMatcher 寻找最近邻近似匹配
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    # lens = min(len(kp1), len(kp2))
    # kp1, desc1 = kp1[:lens], desc1[:lens]
    # kp2, desc2 = kp2[:lens], desc2[:lens]
    raw_matches = flann.knnMatch(desc1, desc2, 2)  # 2

    good = []

    for m, n in raw_matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > 10:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # inline1, inline2 = src_pts[mask.squeeze()], dst_pts[mask.squeeze()]
        inline1 = [src_pts[idx]
                   for idx, data in enumerate(mask.squeeze()) if data == 1]
        inline2 = [dst_pts[idx]
                   for idx, data in enumerate(mask.squeeze()) if data == 1]
        inline1, inline2 = np.stack(
            [inline1], axis=0).squeeze(), np.stack([inline2], axis=0).squeeze()
        E, _ = cv2.findEssentialMat(src_pts, dst_pts, mtx)
        print(E)

        matchesMask = mask.ravel().tolist()

    else:
        print("Not enough matches are found - %d/%d" % (len(good), 10))
        matchesMask = None

    # plot the inline points
    plt.subplot(2, 1, 1)
    plt.scatter(inline1.squeeze()[:, 0], 480-inline1.squeeze()[:, 1])
    plt.subplot(2, 1, 2)
    plt.scatter(inline2.squeeze()[:, 0], 480-inline2.squeeze()[:, 1])
    plt.savefig('2dplot.jpg')

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=(0, 0, 255),
                       matchesMask=matchesMask,
                       flags=2)  # draw only inliers

    vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    # cv2.imshow("", vis)
    cv2.imwrite("face_brisk_bf_ransac_1519.jpg", vis)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    _, ori2_1, loc2_1, _ = cv2.recoverPose(E, src_pts, dst_pts, mtx)
    [M_rot2_1, M_trans2_1] = cameraPoseToExtrinsics(ori2_1, loc2_1)

    ori1_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    loc1_1 = np.array([0, 0, 0])
    [M_rot1_1, M_trans1_1] = cameraPoseToExtrinsics(ori1_1, loc1_1)

    M_camera1 = cameraMatrix(mtx, M_rot1_1, M_trans1_1)
    M_camera2 = cameraMatrix(mtx, M_rot2_1, M_trans2_1)

    tmp = cv2.triangulatePoints(M_camera1, M_camera2, src_pts, dst_pts)
    print(tmp)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = tmp[0] / tmp[3]
    y = tmp[1] / tmp[3]
    z = tmp[2] / tmp[3]
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig('3d.jpg')
    print('debug')
