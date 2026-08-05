clear,clc,clf,close all
warning('off');
addpath("qtfm_2_8\qtfm");
addpath("Denoiser\");
addpath("demof\");

 x0 = double(imread('House.png'));
 
[xN,yN,~]=size(x0); 
BlurDim = 13;  % filter size (squared): 13x13 pixels
filter_type =1;% filter type
%%%***************************************************************
switch filter_type
    case 1 
        ht=ones(BlurDim); ht=ht./sum(ht(:)); %% Uniform/average 13x13 pixels  
    case 2 
        ht = fspecial('disk',BlurDim/2);  ht=ht./sum(ht(:));  % out_of_focus 13x13 pixels 
    case 3    
        ht=fspecial('motion', 17, 135);   ht=ht./sum(ht(:));  %% Motion 13x13 pixels         
    case 4  
        ht=fspecial('gaussian', 13, sqrt(13));   ht=ht./sum(ht(:));  %% Gaussian 13x13 pixels        
end
fsize = round( (BlurDim-1)/2 );  % fsize - size of each side of the filter (6)

[imgB, H] = addblur(ht, x0); 
y_blur = convn(x0, ht, 'valid');  %% blurry image only size 244*244 ;
[sz1_x0, sz2_x0,~] =  size(y_blur);

Ratio_Set = [0.2, 0.3, 0.5, 0.8,1];
ratio = Ratio_Set(5); % ratio of available data
%%%***************************************************************

P = double(rand(sz1_x0,sz2_x0) > (1-ratio));%% random pixel missing
BSNR=30;
sigma = sqrt(var(y_blur(:)) / 10^(BSNR/10));

%%%***************************************************************
y_observed = P.*(y_blur + sigma*randn(size(y_blur)) );
x0_crop = x0(1+fsize:end-fsize, 1+fsize:end-fsize,:);
Mask = zeros(xN,yN);
Mask(1+fsize:end-fsize, 1+fsize:end-fsize) = double(P) ;

y_input = zeros(xN,yN,3);
y_input(1+fsize:end-fsize, 1+fsize:end-fsize,:) =  y_observed  ;
figure(1) ; imshow(P) ;
figure(2) ; imshow(uint8(y_observed)) ;
figure(3) ; imshow(Mask) ;
figure(4) ; imshow(uint8(y_input)) ;

%%%***************************************************************
opts.lambda1 =35; %% QWNNM regularization parameter 
opts.mu1 = 1e-3; %% QWNNM  Lagrange penalty parameter
opts.mu2 =0.01; %% Lagrange penalty parameter
opts.tol = 1e-5;  %% tolerance condition
opts.maxit =150; %% maximum iteration number
%%%***************************************************************
fprintf('**************************************************************\n')
fprintf('***************************************************************\n')
fprintf('Running Please waitting ...\n')
x_out = BAQF_Deblur_Inpaint(opts, x0, y_input, ht, Mask,H); 
fprintf('Running end ...\n')
fprintf('***************************************************************\n')
fprintf('***************************************************************\n')
psnr_out = psnr(x_out./255,x0./255) ;ssim_out=ssim(x_out./255,x0./255);
fprintf('Final estimate PSNR: %4.2f SSIM: %4.4f  \n',  psnr_out,ssim_out); 

figure(5); imshow(uint8(x0),[]); 
title('Original full image','fontsize',13) ;
figure(6); imshow(uint8(y_input),[]); 
title('Observed image','fontsize',13) ;
figure(7); imshow(uint8(x_out),[]); 
title(sprintf('Recovered full image, PSNR: %4.2fdB,SSIM: %4.4f',psnr_out,ssim_out),'fontsize',10);

