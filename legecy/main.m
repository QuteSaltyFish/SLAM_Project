clear;
clc;
close all
%%%���ȶ�ȡ����ͼ����ͳһresize��[512��512]��С
IMG1 = imread('1.jpg');
IMG2 = imread('2.jpg');
IMG3 = imread('3.jpg');
IMG1 = imresize(IMG1,[512,512]);
IMG2 = imresize(IMG2,[512,512]);
IMG3 = imresize(IMG3,[512,512]);

%%%step1:���궨������ڲ�����
mycamera = cameraIntrinsics([564.18636,533.35008],[341.35769,225.86086],[512,512],'RadialDistortion',[0.02970,-0.05793,0],'TangentialDistortion',[-0.00073,0.00306],'Skew',0);

%%%step3:ȥ���䲢ת��Ϊ�Ҷ�ͼ
IMG1 = undistortImage(IMG1,mycamera);
IMG1 = rgb2gray(IMG1);
IMG2 = undistortImage(IMG2,mycamera);
IMG2 = rgb2gray(IMG2);
IMG3 = undistortImage(IMG3,mycamera);
IMG3 = rgb2gray(IMG3);

%%%step4:��SURF����������ƥ��
%%%��ȡ����ͼ����������������������
IMG1_points= detectSURFFeatures(IMG1);
[IMG1_Features,IMG1_points_new] = extractFeatures(IMG1,IMG1_points);
IMG2_points= detectSURFFeatures(IMG2);
[IMG2_Features,IMG2_points_new] = extractFeatures(IMG2,IMG2_points);
%IMG3_points= detectSURFFeatures(IMG3);
%[IMG3_Features,IMG3_points_new] = extractFeatures(IMG3,IMG3_points);
%%%����ƥ��������Ӧ����������루��ֵ0.7ɸѡ��
[index_12,dist12] = matchFeatures(IMG1_Features,IMG2_Features,'Maxratio',0.7); 
%[indexPairs_13,dist13] = matchFeatures(IMG1_Features,IMG3_Features,'Maxratio',0.7);

points_12_IMG1 = IMG1_points_new(index_12(:,1),:);
points_12_IMG2 = IMG2_points_new(index_12(:,2),:);
%matchedPts1_13 = IMG1_points_new(indexPairs_13(:,1),:);
%matchedPts3_13 = IMG3_points_new(indexPairs_13(:,2),:);

%%%��ʾ����ƥ��֮��ĵ㼯
figure;
showMatchedFeatures(IMG1,IMG2,points_12_IMG1,points_12_IMG2,'montage');
title('1-2ƥ��㼯');
%%%%���������ٴ���ʾ
%[t,inlier1,inlier2] = estimateGeometricTransform(points_12_IMG1,matchedPts2_12,'affine','MaxDistance',30);

%figure;
%showMatchedFeatures(IMG1,IMG2,inlier1,inlier2,'montage');
%title('1-2�ڵ�ƥ��㼯');


%%%%step5:���1-2ƥ���Ӧ�ı�������ͻ�������
Num = 3000; % ����ɵ���
% ��RANSAC����������Ͷ�Ӧ�ڵ�
[F,inliers] = estimateFundamentalMatrix(points_12_IMG1,points_12_IMG2,'Method','RANSAC','NumTrials',Num,'DistanceThreshold',1e-4);
%[E,inliers] = estimateEssentialMatrix(matchedPts1_12,matchedPts2_12,camPara);

% ȡ���ڵ�
inliers_1 = points_12_IMG1(inliers);
inliers_2 = points_12_IMG2(inliers);
% �����inliers�ý�ȥ����һ�鱾������
[E,~] = estimateEssentialMatrix(inliers_1,inliers_2,mycamera);
E
%%%�����ڵ��Ӧ��ƥ�����
figure;
showMatchedFeatures(IMG1,IMG2,inliers_1,inliers_2,'montage');
title('RANSAC���������Ӧ1-2�ڵ�ƥ��');


%%%%step6:�ӱ��������еõ�2���1��ת�����ƽ�ƾ���
%%%%�ȹ������λ�ã�����+���룩
[ori2_1,loc2_1] = relativeCameraPose(E,mycamera,inliers_1,inliers_2);
%%%%�����λ�ù��ƾ���
%%%%2���1�ľ���
[M_rot2_1,M_trans2_1] = cameraPoseToExtrinsics(ori2_1,loc2_1)
%%%%1��Ա���ľ�����һ�����ã�
ori1_1 = [1,0,0;0,1,0;0,0,1];
loc1_1 = [0,0,0];
[M_rot1_1,M_trans1_1] = cameraPoseToExtrinsics(ori1_1,loc1_1); 


%%%%%step7:���ǻ���ά�����ع�
%%%%%���������������
M_camera1 = cameraMatrix(mycamera,M_rot1_1,M_trans1_1);
M_camera2 = cameraMatrix(mycamera,M_rot2_1,M_trans2_1);
%%%%%%���������������ά������
[points_3D,err] = triangulate(points_12_IMG1,points_12_IMG2,M_camera1,M_camera2);



%%%%%step8:����������ͼ��2D-3D��Ӧ��ϵ
%%%%%������������һ�ŵ�ƥ�����������step4
IMG3_points= detectSURFFeatures(IMG3);
[IMG3_Features,IMG3_points_new] = extractFeatures(IMG3,IMG3_points);
[index_13,dist13] = matchFeatures(IMG1_Features,IMG3_Features,'Maxratio',0.7);
points_13_IMG1 = IMG1_points_new(index_13(:,1),:);
points_13_IMG3 = IMG3_points_new(index_13(:,2),:);
%%%%%���ӳ�����
[asd,index12_1,index13_1] = intersect(index_12(:,1),index_13(:,1));
%%%%%�����ά����ϵ�µĵ㣨������ͼ��
points_3D_IMG3 = points_3D(index12_1,:);
%%%%%����ά����ϵ�µĶ�Ӧ�㣨������ͼ��
points_2D_IMG3 = points_13_IMG3(index13_1,:).Location;

%%%%%step9:ʹ��RANSAC��M-estimatorʵ�ֵ������ӽǵ����λ����̬���㡣
%%%%%�����������������άλ��
[ori3,loc3] = estimateWorldCameraPose(points_2D_IMG3,points_3D_IMG3,mycamera);

%%%%%�������ת������ƽ�ƾ���
[M_rot3_1,M_trans3_1] = cameraPoseToExtrinsics(ori3,loc3)


%%%%%��һ�ַ�����֤��
%%%%%ֱ����1-3ƥ�����
[E_13,inliers] = estimateEssentialMatrix(points_13_IMG1,points_13_IMG3,mycamera);
inliers1 = points_13_IMG1(inliers);
inliers3 = points_13_IMG3(inliers);
[ori3_1,loc3_1] = relativeCameraPose(E_13,mycamera,inliers1,inliers3);
[M_rot3_1_new,M_trans3_1_new] = cameraPoseToExtrinsics(ori3_1,loc3_1);
figure;
plot3(points_3D(:,1),points_3D(:,2),points_3D(:,3),'.',M_trans3_1(1),M_trans3_1(2),M_trans3_1(3),'*',M_trans2_1(1),M_trans2_1(2),M_trans2_1(3),'*',M_trans1_1(1),M_trans1_1(2),M_trans1_1(3),'*');
grid on;










