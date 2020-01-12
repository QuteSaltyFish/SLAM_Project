clear;
clc;
close all
load('camera.mat')
%%%将图片压缩到480x680
I1 = imread('1.jpg');
I2 = imread('2.jpg');
I3 = imread('3.jpg');
I1 = imresize(I1,[480,680]);
I2 = imresize(I2,[480,680]);
I3 = imresize(I3,[480,680]);

%%%step1:标定相机数据
mycamera = cameraParams;

%%%step3:去除相机畸变
I1 = undistortImage(I1,mycamera);
% I1 = rgb2gray(I1);
I2 = undistortImage(I2,mycamera);
% I2 = rgb2gray(I2);
I3 = undistortImage(I3,mycamera);
% I3 = rgb2gray(I3);

%%%step4:使用SURF对第一第二张图进行特征点检测
SURF_Pt_1 = detectSURFFeatures(rgb2gray(I1));
SURF_Pt_2 = detectSURFFeatures(rgb2gray(I2));

[features_1,validPoints_1] = extractFeatures(rgb2gray(I1),SURF_Pt_1);
[features_2,validPoints_2] = extractFeatures(rgb2gray(I2),SURF_Pt_2);

%按照0.7的比例筛选
indexPairs = matchFeatures(features_1, features_2,'MaxRatio',0.7,'Unique',true) ;
matchedPoints1_12 = validPoints_1(indexPairs(:,1));
matchedPoints2_12 = validPoints_2(indexPairs(:,2));

%plot the matched points
figure; 
showMatchedFeatures(I1,I2,matchedPoints1_12,matchedPoints2_12,'montage','PlotOptions',{'ro','go','y--'});
legend('matched points 1','matched points 2');

%%%step5:使用RANSEC求解Fundamental Matrix
[F, inliers] = estimateFundamentalMatrix(matchedPoints1_12,matchedPoints2_12,'Method','RANSAC');
inlierPoints1_12 = matchedPoints1_12(inliers,:);
inlierPoints2_12 = matchedPoints2_12(inliers,:);
figure;
showMatchedFeatures(I1, I2, inlierPoints1_12, inlierPoints2_12,'montage','PlotOptions',{'ro','go','y--'});
title('Point matches after outliers were removed');

%重新把inliner拿去重新算一遍Fundamental Matrix
[F,inliers] = estimateFundamentalMatrix(inlierPoints1_12, inlierPoints2_12, 'Method', 'RANSAC');

% [E, epipolarInliers] = estimateEssentialMatrix(...
%     matchedPoints1, matchedPoints2, mycamera, 'Confidence', 99.99);
% % Find epipolar inliers
% inlierPoints1 = matchedPoints1(epipolarInliers, :);
% inlierPoints2 = matchedPoints2(epipolarInliers, :);

%%%step6:从本质矩阵中分解R,并对多个解进行筛选得到唯一解
Ori1 = [1,0,0; 0,1,0; 0,0,1];
Loc1 = [0,0,0];
[M_rot1, M_trans1] = cameraPoseToExtrinsics(Ori1, Loc1);

[reOri_12, reLoc_12] = relativeCameraPose(F, mycamera, inlierPoints1_12, inlierPoints2_12);
[M_rot2, M_trans2] = cameraPoseToExtrinsics(reOri_12, reLoc_12);

Cam1 = cameraMatrix(cameraParams, eye(3), [0 0 0]);

Cam2 = cameraMatrix(mycamera,M_rot2,M_trans2);

%%%step7利用三角化进行三维点云重构
worldPoint = triangulate(matchedPoints1_12,matchedPoints2_12,Cam1,Cam2);


% %%%step8将两张图中的一张与第三张进行特征匹配
% %提取特征点,并且特征匹配
SURF_Pt_3 = detectSURFFeatures(rgb2gray(I3));
[features_3,validPoints_3] = extractFeatures(rgb2gray(I3),SURF_Pt_3);
indexPairs2 = matchFeatures(features_1, features_3, 'MaxRatio', 0.7, 'Unique', true);
matchedPoints1_13 = validPoints_1(indexPairs2(:,1));
matchedPoints3_13 = validPoints_3(indexPairs2(:,2));

%找到两次匹配相同的在第一张图上的点
[asd,index12_1,index13_1] = intersect(indexPairs(:,1),indexPairs2(:,1));
worldPoint_i3 = worldPoint(index12_1,:);
pixelPoint_i3 = matchedPoints3_13(index13_1,:).Location;

% %估计相机位置
[reOri_13, reLoc_13] = estimateWorldCameraPose(pixelPoint_i3,worldPoint_i3,mycamera);
[M_rot3, M_trans3] = cameraPoseToExtrinsics(reOri_13, reLoc_13);
Cam3 = cameraMatrix(mycamera,M_rot3,M_trans3);
figure;
showMatchedFeatures(I1, I3, matchedPoints1_12(index12_1,:), pixelPoint_i3,'montage','PlotOptions',{'ro','go','y--'});
title('I1 and I3');

%%% prepare the parameters used for bundle adjustment
vSet = viewSet;
vSet = addView(vSet, 1,'Points',validPoints_1,'Orientation',...
    M_rot1,'Location',M_trans1);
vSet = addView(vSet, 2,'Points',validPoints_2,'Orientation',...
    M_rot2,'Location',M_trans2);
vSet = addConnection(vSet,1,2,'Matches',indexPairs);
vSet = addView(vSet, 3,'Points',validPoints_3,'Orientation',...
    M_rot3,'Location',M_trans3);
vSet = addConnection(vSet,1,3,'Matches',indexPairs2);

tracks = findTracks(vSet);
cameraPoses = poses(vSet);

% ViewId = {1; 2; 3};
% Orientation = {eye(3); reOri_12; reOri_13};
% Location = {Loc1; reLoc_12; reLoc_13};
% CameraPoses = table(ViewId, Orientation, Location);


numPixels = size(I1, 1) * size(I1, 2);
allColors = reshape(I1, [numPixels, 3]);
tmp1 = matchedPoints1_12.Location(:,2);
tmp2 = matchedPoints1_12.Location(:,1);
colorIdx = sub2ind([size(I1, 1), size(I1, 2)], round(tmp1(index12_1,:)), ...
    round(tmp2(index12_1,:)));
color = allColors(colorIdx, :);

[xyzPoints,errors] = triangulateMultiview(tracks,cameraPoses,cameraParams);

% z = xyzPoints(:,3);
% idx = errors < 5 & z > 0 & z < 20;
% pcshow(xyzPoints(idx, :),'VerticalAxis','y','VerticalAxisDir','down','MarkerSize',30);
% hold on
% plotCamera(cameraPoses, 'Size', 0.1);
% hold off


[xyzRefinedPoints,refinedPoses] = ...
    bundleAdjustment(xyzPoints,tracks,cameraPoses,cameraParams);
pcshow(xyzRefinedPoints,'VerticalAxis','y','VerticalAxisDir',...
    'down','MarkerSize',45);
hold on
plotCamera(cameraPoses, 'Size', 0.1)
hold off
% % Create the point cloud
% ptCloud = pointCloud(worldPoint_i3, 'Color', color);


% % Visualize the camera locations and orientations
% cameraSize = 0.3;
% figure
% plotCamera('Size', cameraSize, 'Color', 'r', 'Label', '1', 'Opacity', 0);
% hold on
% grid on
% plotCamera('Location', reLoc_12, 'Orientation', reOri_12, 'Size', cameraSize, ...
%     'Color', 'b', 'Label', '2', 'Opacity', 0);
% plotCamera('Location', reLoc_13, 'Orientation', reOri_13, 'Size', cameraSize, ...
%     'Color', 'g', 'Label', '3', 'Opacity', 0);

% % Visualize the point cloud
% pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
%     'MarkerSize', 45);




% % Rotate and zoom the plot
% camorbit(0, -30);
% camzoom(1.5);

% % Label the axes
% xlabel('x-axis');
% ylabel('y-axis');
% zlabel('z-axis');

% title('Up to Scale Reconstruction of the Scene');
