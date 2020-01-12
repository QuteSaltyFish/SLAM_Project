clear;
clc;
close all
load('camera.mat')
%%%将图片压缩到480x680
IMG1 = imread('1.jpg');
IMG2 = imread('2.jpg');
IMG3 = imread('3.jpg');
IMG1 = imresize(IMG1,[480,680]);
IMG2 = imresize(IMG2,[480,680]);
IMG3 = imresize(IMG3,[480,680]);

%%%step1:标定相机数据
mycamera = cameraParams;

%%%step3:去畸变并转化为灰度图
IMG1 = undistortImage(IMG1,mycamera);
IMG1 = rgb2gray(IMG1);
IMG2 = undistortImage(IMG2,mycamera);
IMG2 = rgb2gray(IMG2);
IMG3 = undistortImage(IMG3,mycamera);
IMG3 = rgb2gray(IMG3);

%%%step4:用SURF进行特征点匹配
%%%提取两张图的特征点与特征描述算子
IMG1_points= detectSURFFeatures(IMG1);
[IMG1_Features,IMG1_points_new] = extractFeatures(IMG1,IMG1_points);
IMG2_points= detectSURFFeatures(IMG2);
[IMG2_Features,IMG2_points_new] = extractFeatures(IMG2,IMG2_points);
%IMG3_points= detectSURFFeatures(IMG3);
%[IMG3_Features,IMG3_points_new] = extractFeatures(IMG3,IMG3_points);
%%%返回匹配特征对应的索引与距离（比值0.7筛选）
[index_12,dist12] = matchFeatures(IMG1_Features,IMG2_Features,'Maxratio',0.7); 
%[indexPairs_13,dist13] = matchFeatures(IMG1_Features,IMG3_Features,'Maxratio',0.7);

points_12_IMG1 = IMG1_points_new(index_12(:,1),:);
points_12_IMG2 = IMG2_points_new(index_12(:,2),:);
%matchedPts1_13 = IMG1_points_new(indexPairs_13(:,1),:);
%matchedPts3_13 = IMG3_points_new(indexPairs_13(:,2),:);

%%%显示初步匹配之后的点集
figure;
showMatchedFeatures(IMG1,IMG2,points_12_IMG1,points_12_IMG2,'montage');
title('1-2匹配点集');
%%%%消除外点后再次显示
%[t,inlier1,inlier2] = estimateGeometricTransform(points_12_IMG1,matchedPts2_12,'affine','MaxDistance',30);

%figure;
%showMatchedFeatures(IMG1,IMG2,inlier1,inlier2,'montage');
%title('1-2内点匹配点集');


%%%%step5:求解1-2匹配对应的本征矩阵和基础矩阵
Num = 3000; % 随机采点数
% 用RANSAC求解基础矩阵和对应内点
[F,inliers] = estimateFundamentalMatrix(points_12_IMG1,points_12_IMG2,'Method','RANSAC','NumTrials',Num,'DistanceThreshold',1e-4);
%[E,inliers] = estimateEssentialMatrix(matchedPts1_12,matchedPts2_12,camPara);

% 取出内点
inliers_1 = points_12_IMG1(inliers);
inliers_2 = points_12_IMG2(inliers);
% 这里把inliers拿进去再算一遍本征矩阵
[E,~] = estimateEssentialMatrix(inliers_1,inliers_2,mycamera);
E
%%%画出内点对应的匹配情况
figure;
showMatchedFeatures(IMG1,IMG2,inliers_1,inliers_2,'montage');
title('RANSAC本征矩阵对应1-2内点匹配');


%%%%step6:从本征矩阵中得到2相对1旋转矩阵和平移矩阵
%%%%先估计相对位置（方向+距离）
[ori2_1,loc2_1] = relativeCameraPose(E,mycamera,inliers_1,inliers_2);
%%%%由相对位置估计矩阵
%%%%2相对1的矩阵
[M_rot2_1,M_trans2_1] = cameraPoseToExtrinsics(ori2_1,loc2_1)
%%%%1相对本身的矩阵（下一步有用）
ori1_1 = [1,0,0;0,1,0;0,0,1];
loc1_1 = [0,0,0];
[M_rot1_1,M_trans1_1] = cameraPoseToExtrinsics(ori1_1,loc1_1); 


%%%%%step7:三角化三维点云重构
%%%%%先求两个相机矩阵
M_camera1 = cameraMatrix(mycamera,M_rot1_1,M_trans1_1);
M_camera2 = cameraMatrix(mycamera,M_rot2_1,M_trans2_1);
%%%%%%根据相机矩阵求三维点坐标
[points_3D,err] = triangulate(points_12_IMG1,points_12_IMG2,M_camera1,M_camera2);



%%%%%step8:构建第三张图的2D-3D对应关系
%%%%%先求第三张与第一张的匹配情况，仿照step4
IMG3_points= detectSURFFeatures(IMG3);
[IMG3_Features,IMG3_points_new] = extractFeatures(IMG3,IMG3_points);
[index_13,dist13] = matchFeatures(IMG1_Features,IMG3_Features,'Maxratio',0.7);
points_13_IMG1 = IMG1_points_new(index_13(:,1),:);
points_13_IMG3 = IMG3_points_new(index_13(:,2),:);
%%%%%求解映射情况
[asd,index12_1,index13_1] = intersect(index_12(:,1),index_13(:,1));
%%%%%求解三维坐标系下的点（第三张图）
points_3D_IMG3 = points_3D(index12_1,:);
%%%%%求解二维坐标系下的对应点（第三张图）
points_2D_IMG3 = points_13_IMG3(index13_1,:).Location;

%%%%%step9:使用RANSAC或M-estimator实现第三个视角的相机位置姿态估算。
%%%%%先求解第三个相机的三维位置
[ori3,loc3] = estimateWorldCameraPose(points_2D_IMG3,points_3D_IMG3,mycamera);

%%%%%再求解旋转矩阵与平移矩阵
[M_rot3_1,M_trans3_1] = cameraPoseToExtrinsics(ori3,loc3)


%%%%%另一种方法验证：
%%%%%直接由1-3匹配求解
[E_13,inliers] = estimateEssentialMatrix(points_13_IMG1,points_13_IMG3,mycamera);
inliers1 = points_13_IMG1(inliers);
inliers3 = points_13_IMG3(inliers);
[ori3_1,loc3_1] = relativeCameraPose(E_13,mycamera,inliers1,inliers3);
[M_rot3_1_new,M_trans3_1_new] = cameraPoseToExtrinsics(ori3_1,loc3_1);
figure;
plot3(points_3D(:,1),points_3D(:,2),points_3D(:,3),'.',M_trans3_1(1),M_trans3_1(2),M_trans3_1(3),'*',M_trans2_1(1),M_trans2_1(2),M_trans2_1(3),'*',M_trans1_1(1),M_trans1_1(2),M_trans1_1(3),'*');
grid on;