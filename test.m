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
matchedPoints1 = validPoints_1(indexPairs(:,1));
matchedPoints2 = validPoints_2(indexPairs(:,2));

%plot the matched points
figure; 
showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage','PlotOptions',{'ro','go','y--'});
legend('matched points 1','matched points 2');

%%%step5:使用RANSEC求解Fundamental Matrix
[F, inliers] = estimateFundamentalMatrix(matchedPoints1,matchedPoints2,'Method','RANSAC');
inlierPoints1 = matchedPoints1(inliers,:);
inlierPoints2 = matchedPoints2(inliers,:);
figure;
showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2,'montage','PlotOptions',{'ro','go','y--'});
title('Point matches after outliers were removed');

% Estimate the fundamental matrix
% [E, epipolarInliers] = estimateEssentialMatrix(...
%     matchedPoints1, matchedPoints2, cameraParams, 'Confidence', 99.99);

% % Find epipolar inliers
% inlierPoints1 = matchedPoints1(epipolarInliers, :);
% inlierPoints2 = matchedPoints2(epipolarInliers, :);

% Display inlier matches
figure
showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2);
title('Epipolar Inliers');

[orient, loc] = relativeCameraPose(F, cameraParams, inlierPoints1, inlierPoints2);



% Compute the camera matrices for each position of the camera
% The first camera is at the origin looking along the Z-axis. Thus, its
% rotation matrix is identity, and its translation vector is 0.
camMatrix1 = cameraMatrix(cameraParams, eye(3), [0 0 0]);

% Compute extrinsics of the second camera
[R, t] = cameraPoseToExtrinsics(orient, loc);
camMatrix2 = cameraMatrix(cameraParams, R, t);

% Compute the 3-D points
points3D = triangulate(matchedPoints1, matchedPoints2, camMatrix1, camMatrix2);
figure;
plot3(points3D(:,1),points3D(:,2),points3D(:,3),'.');
title('3d point Cloud from img1 and img2')


% Get the color of each reconstructed point
numPixels = size(I1, 1) * size(I1, 2);
allColors = reshape(I1, [numPixels, 3]);
colorIdx = sub2ind([size(I1, 1), size(I1, 2)], round(matchedPoints1.Location(:,2)), ...
    round(matchedPoints1.Location(:, 1)));
color = allColors(colorIdx, :);

% Create the point cloud
ptCloud = pointCloud(points3D, 'Color', color);


% Visualize the camera locations and orientations
cameraSize = 0.3;
figure
plotCamera('Size', cameraSize, 'Color', 'r', 'Label', '1', 'Opacity', 0);
hold on
grid on
plotCamera('Location', loc, 'Orientation', orient, 'Size', cameraSize, ...
    'Color', 'b', 'Label', '2', 'Opacity', 0);

% Visualize the point cloud
pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);

% Rotate and zoom the plot
camorbit(0, -30);
camzoom(1.5);

% Label the axes
xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis')

title('Up to Scale Reconstruction of the Scene');