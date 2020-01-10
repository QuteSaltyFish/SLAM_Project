import numpy as np
import cv2


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
        retval, rotations, translations, normals = cv2.decomposeHomographyMat(
            H, mtx)
        inline1, inline2 = src_pts[mask.squeeze()], dst_pts[mask.squeeze()]
        E, _ = cv2.findEssentialMat(inline1, inline2, mtx)
        print(E)
        _, R, t, _ = cv2.recoverPose(E, inline1, inline2, mtx)
        print(R, t)
        matchesMask = mask.ravel().tolist()

    else:
        print("Not enough matches are found - %d/%d" % (len(good), 10))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=(0, 0, 255),
                       matchesMask=matchesMask,
                       flags=2)  # draw only inliers

    vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    # cv2.imshow("", vis)
    cv2.imwrite("face_brisk_bf_ransac_1519.jpg", vis)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
