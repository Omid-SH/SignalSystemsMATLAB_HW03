%% Part 1
% Finding edges of a picture

%% 1.1
clear; clc

% Sobel kernels
gx = [-1 0 1; -2 0 2; -1 0 1];
gy = gx';

% This commented part is for RGB sobel answer

% % plotting
% figure(1)
% %pic = double(importdata('pic_sobel_1.png'));
% pic = double(importdata('pic_sobel_2.jpg'));
% subplot(1,2,1);
% image(uint8(pic));
% 
% % calculaintg sobel answer for each color
% for k = 1:3
%     picx = conv2(pic(:,:,k),gx,'same');
%     picy = conv2(pic(:,:,k),gy,'same');
%     pic(:,:,k) = sqrt(picx.^2 + picy.^2);
% end
% subplot(1,2,2);
% image(uint8(pic));

% plotting
figure(1)
pic = rgb2gray(importdata('pic_sobel_1.png'));
%pic = rgb2gray(importdata('pic_sobel_2.jpg'));

% use this picture to compare execution time of sobel with kirsch operator 
%pic = rgb2gray(importdata('pic_kirsch_2.jpg')); 

subplot(1,2,1);
imshow(uint8(pic));

tic
% calculaintg sobel answer
picx = conv2(pic,gx,'same');
picy = conv2(pic,gy,'same');
pic = sqrt(picx.^2 + picy.^2);
toc

subplot(1,2,2);
imshow(uint8(pic));


%% 1.2
clear; clc

% kirsch 8 kernels
g = zeros(3,3,8);
g(:,:,1) = [5,5,5; -3,0,-3; -3,-3,-3];
g(:,:,2) = [5,5,-3; 5,0,-3; -3,-3,-3];
g(:,:,3) = [5,-3,-3; 5,0,-3; 5,-3,-3];
g(:,:,4) = [-3,-3,-3; 5,0,-3; 5,5,-3];
g(:,:,5) = [-3,-3,-3; -3,0,-3; 5,5,5];
g(:,:,6) = [-3,-3,-3; -3,0,5; -3,5,5];
g(:,:,7) = [-3,-3,5; -3,0,5; -3,-3,5];
g(:,:,8) = [-3,5,5; -3,0,5; -3,-3,-3];

% This commented part is for RGB kirsch answer becuase my both pictures are
% somehow gray, you can't see noticeable difference between answers of this
% part and next one

% % ploting
% figure(1)
% %pic = double(importdata('pic_kirsch_1.jpg'));
% pic = double(importdata('pic_kirsch_2.jpg'));
% subplot(1,2,1);
% image(uint8(pic));
% 
% pictmp = zeros(size(pic,1),size(pic,2),8);
% 
% % calculating kirsch answer
% for k = 1:3
%     for i = 1:8
%         pictmp(:,:,i) = conv2(pic(:,:,k),g(:,:,i),'same');
%     end
%     pic(:,:,k) = max(pictmp,[],3);
% end
% 
% subplot(1,2,2);
% image(uint8(pic));

% ploting
figure(1)
%pic = rgb2gray(importdata('pic_kirsch_1.jpg'));
pic = rgb2gray(importdata('pic_kirsch_2.jpg'));
subplot(1,2,1);
imshow(uint8(pic));

pictmp = zeros(size(pic,1),size(pic,2),8);

tic
% calculating kirsch answer
for k = 1:8
    pictmp(:,:,k) = conv2(pic,g(:,:,k),'same');
end
pic = max(pictmp,[],3);
toc

subplot(1,2,2);
imshow(uint8(pic));

%% Part2

%% code using imfindcircles
clear; clc
pic = rgb2gray(imread('circles.jpg'));

% Sobel filter
gx = [-1 0 1; -2 0 2; -1 0 1];
gy = gx';
pic = double(pic);
picx = conv2(pic,gx,'same');
picy = conv2(pic,gy,'same');
pic = sqrt(picx.^2 + picy.^2);

% denoising using Gaussian Filter and sharpen filter
pic = Gaussian_Filter(pic , 15, 3);
pic = imsharpen(pic);
pic = uint8(pic);
imshow(pic);

[centers,rad] = imfindcircles(pic,[15 30]);
viscircles(centers, rad,'Color','b');
title(['Number of circles = ',num2str(length(centers))]);

%% Hand defined circle counter function
clear; clc
pic = rgb2gray(imread('circles.jpg'));

% Sobel filter
gx = [-1 0 1; -2 0 2; -1 0 1];
gy = gx';
pic = double(pic);
picx = conv2(pic,gx,'same');
picy = conv2(pic,gy,'same');
pic = sqrt(picx.^2 + picy.^2);
pic = imsharpen(pic);

[N ,centers] = FindCircle(pic , 22 , 24 , 2.4*10^4);
pic = uint8(pic);
imshow(pic);

viscircles(centers, 1.*ones(1,length(centers)),'Color','b');
title(['Number of circles is ',num2str(N)]);


%% Part3

clear; clc
pic1 = imread('pic1.png');
pic2 = imread('pic2.png');
pic3 = ifft2(abs(fft2(pic2)).*exp(1i*angle(fft2(pic1))));

figure(1)
imshow(uint8(pic3));

%% Part4

%% a
clear; clc

I = imread('pic1.png');

J1 = imnoise(I,'salt & pepper',0.02);
J2 = imnoise(I,'gaussian',0,0.01);
J3 = imnoise(I,'poisson');
J4 = imnoise(I,'speckle',0.04);

figure(1)

subplot(2,2,1)
imshow(J1)
title('salt & pepper')

subplot(2,2,2)
imshow(J2)
title('guassian')

subplot(2,2,3)
imshow(J3)
title('poisson')

subplot(2,2,4)
imshow(J4)
title('speckle')

% b
% No code

% c.1
% Guassian_Filter

% c.2
% Median_Filter

% c.3

%%

% Gaussian filter for salt & pepper
figure('Name' , 'Gaussian Filter salt & pepper output')
subplot(1,2,1)
imshow(J1)
title('salt & pepper')

for Ks = 3:2:9
    for sigma = 0.6:0.2:1.4
        subplot(1,2,2)
        tmp = Gaussian_Filter(double(J1),Ks,sigma);
        imshow(uint8(tmp));
        title(['Output of Gaussian Filter (Ks =  ',num2str(Ks),' \sigma = ',num2str(sigma),')']);
        pause(0.1)
    end
end

% Gaussian filter for Gaussian
figure('Name' , 'Gaussian Filter Gaussian output')
subplot(1,2,1)
imshow(J2)
title('Gaussian')

for Ks = 3:2:9
    for sigma = 0.6:0.2:1.4
        subplot(1,2,2)
        tmp = Gaussian_Filter(double(J2),Ks,sigma);
        imshow(uint8(tmp));
        title(['Output of Gaussian Filter (Ks =  ',num2str(Ks),' \sigma = ',num2str(sigma),')']);
        pause(0.1)
    end
end

% Gaussian filter for Poisson
figure('Name' , 'Gaussian Filter Poisson output')
subplot(1,2,1)
imshow(J3)
title('Poisson')

for Ks = 3:2:9
    for sigma = 0.6:0.2:1.4
        subplot(1,2,2)
        tmp = Gaussian_Filter(double(J3),Ks,sigma);
        imshow(uint8(tmp));
        title(['Output of Gaussian Filter (Ks =  ',num2str(Ks),' \sigma = ',num2str(sigma),')']);
        pause(0.1)
    end
end

% Gaussian filter for Speckle
figure('Name' , 'Gaussian Filter Speckle output')
subplot(1,2,1)
imshow(J4)
title('Speckle')

for Ks = 3:2:9
    for sigma = 0.6:0.2:1.4
        subplot(1,2,2)
        tmp = Gaussian_Filter(double(J4),Ks,sigma);
        imshow(uint8(tmp));
        title(['Output of Gaussian Filter (Ks =  ',num2str(Ks),' \sigma = ',num2str(sigma),')']);
        pause(0.1)
    end
end

%% 

% Median filter for salt & pepper
figure('Name' , 'Median Filter salt & pepper output')
subplot(1,2,1)
imshow(J1)
title('salt & pepper')

for Ks = 3:2:9
    subplot(1,2,2)
    tmp = Median_Filter(double(J1),Ks);
    imshow(uint8(tmp));
    title(['Output of Median Filter (Ks =  ',num2str(Ks),')']);
    pause(0.1)
end

% Median filter for Gaussian
figure('Name' , 'Median Filter Gaussian output')
subplot(1,2,1)
imshow(J2)
title('Gaussian')

for Ks = 3:2:9
    subplot(1,2,2)
    tmp = Median_Filter(double(J2),Ks);
    imshow(uint8(tmp));
    title(['Output of Median Filter (Ks =  ',num2str(Ks),')']);
    pause(0.1)
end

% Median filter for Poisson
figure('Name' , 'Median Filter Poisson output')
subplot(1,2,1)
imshow(J3)
title('Poisson')

for Ks = 3:2:9
    subplot(1,2,2)
    tmp = Median_Filter(double(J3),Ks);
    imshow(uint8(tmp));
    title(['Output of Median Filter (Ks =  ',num2str(Ks),')']);
    pause(0.1)
end

% Median filter for Speckle
figure('Name' , 'Median Filter Speckle output')
subplot(1,2,1)
imshow(J4)
title('Speckle')

for Ks = 3:2:9
    subplot(1,2,2)
    tmp = Median_Filter(double(J4),Ks);
    imshow(uint8(tmp));
    title(['Output of Median Filter (Ks =  ',num2str(Ks),')']);
    pause(0.1)
end

%% c.4

%% Part5

%% This fmri movement detection code is completly available at matlab docs
clear; clc
load('fmri.mat')
moving = image(:,:,1);
fixed = image(:,:,2);
h = imregcorr(moving,fixed);

% plotting orginal pictures
figure(1)
subplot(2,1,1)
imshowpair(moving,fixed,'Scaling','independent','method','montage');
title('moving and fixed pictures ( image(:,:,1) vs image(:,:,2) )');

% set optimizer's fitures
[optimizer, metric] = imregconfig('multimodal');
optimizer.InitialRadius = 0.001;
optimizer.Epsilon = 1e-7;
optimizer.GrowthFactor = 1.001;
optimizer.MaximumIterations = 1000;

% find best ratation and movement
tform = imregtform(moving, fixed, 'affine', optimizer, metric);
movingRegistered = imwarp(moving,tform,'OutputView',imref2d(size(fixed)));

% ploting answer
subplot(2,1,2)
imshowpair(fixed, movingRegistered,'Scaling','independent','method','montage')
title('image(:,:,1) vs image(:,:,2) after moving and rotating image(:,:,1)');

% calculating angle of rotatin, x, y and final value of corolation of two
% pictures
[x, y] = transformPointsForward(tform, [0 1], [0 0]); 
dx = x(2) - x(1); 
dy = y(2) - y(1); 
thet = (180/pi) * atan2(dy, dx) ;
x = x(1);
y = y(1);
max_corolation = corr2(movingRegistered,fixed);
fprintf('X = %d \nY = %d \nTheta = %d \nMax Corolation = %d \n',x,y,thet,max_corolation);

%% Implementing some parts with for loops :)
% running this part of code takes a minute because I set it's precision 0.1

clear; 
load('fmri.mat')

moving = image(:,:,1);
fixed = image(:,:,2);

% we have answers from previous part that helps us for deciding better
% bands for a,b and ththa
max_corolation = 0;
x = 0;
y = 0;
thet = 0;
moving_pic_ans = moving;
for a = 14:0.1:17
    for b = -12:0.1:-9
        for theta = -22:0.1:-18

        tform = affine2d([cosd(theta) -sind(theta) 0; ...
                          sind(theta)  cosd(theta) 0; ...
                          a b 1]);
        tmpmoving= imwarp(moving,tform,'OutputView',imref2d(size(fixed)));
    
        if(corr2(tmpmoving,fixed) > max_corolation)
            max_corolation = corr2(tmpmoving,fixed);
            x = a;
            y = b;
            thet = theta;
            moving_pic_ans = tmpmoving;
        end   

        end
    end
end

fprintf('X = %d \nY = %d \nTheta = %d \nMax Corolation = %d \n',x,y,thet,max_corolation);
figure(1)
imshowpair(fixed, moving_pic_ans,'Scaling','independent')
title(['Final state of pictures after moving ( corolation = ',num2str(max_corolation),' )']);



