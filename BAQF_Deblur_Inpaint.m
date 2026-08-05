function x = BAQF_Deblur_Inpaint(opts, x0, y, ker,  M,H) %% Pixel value ranges in [0,255]
lambda1 = opts.lambda1 ;  
mu1 = opts.mu1   ;  
mu2 = opts.mu2   ;  
tol = opts.tol  ;   
rho=1.001;
maxit = opts.maxit ; 
x=double2q(y);
d1    =       double2q(zeros(size(y))); 
d2    =       double2q(zeros(size(y)));  
A_      =       fft2(H);
AtA_    =       conj(A_).*A_;
                
%%%***************************************************************
q2double = @(X) double2q(X, 'inverse');
y=double2q(y);
MTy = M.*y ;
% set parameters
nSig = lambda1*8.5;                        
Par   = QWNNM_ParSet(nSig); 
par.c         =   1.7*sqrt(2); 
par.delta     =   0.1;  
Par.Innerloop           =       1;
Par.patsize             =       6;
Par.patnum              =       155;      
Par.lamada              =       0.334;      
Par.Iter                =       5;
imgO=x0;
[Height, Width, Depth]  = size(imgO);
TotalPatNum = (Height-Par.patsize+1)*(Width-Par.patsize+1);
Dim = Par.patsize*Par.patsize;
[Neighbor_arr, Num_arr, Self_arr] =	QWNNM_NeighborIndex(imgO, Par);
 NL_mat = zeros(Par.patnum,length(Num_arr));
 CurPat = zeros(Dim, TotalPatNum);
%%%***************************************************************
%%%***************************************************************
PSNR_out = [] ;    ERROR_out = [] ; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%    

%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%% Main function
for nstep=1:maxit 
%%%%%%%%%%%%%%%%%%%%%%%%%%% 
D_      =       mu2 .* AtA_ + mu1;    
%%% z1-subproblem 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_Img =q2double(x+d1/mu1);
    E_Img = N_Img;
    E_Img = E_Img + Par.delta*(N_Img - E_Img);
    [CurPat, Sigma_arr] = QWNNM_Im2Patch(E_Img, N_Img, Par);
    NL_mat = QWNNM_Block_matching(CurPat, Par, Neighbor_arr, Num_arr, Self_arr);
    [EPat, W] = QWNNM_PatEstimation(NL_mat, Self_arr, Sigma_arr, CurPat, Par);
    E_Img = QWNNM_Patch2Im(EPat, W, Par.patsize, Height, Width, Depth);

z1=double2q(E_Img);
%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%% z3-subproblem (MASK OPERATION)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_prev = x ;
Ax=blurq(x,ker);
z2 = (mu2*Ax+d2+MTy)./(mu2 +M);   
%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=ifft2((mu1.*fft2(z1-d1/mu1)+mu2.*real(fft2(H)).*fft2(z2-d2/mu2))./D_);
x_real=q2double(real(x));
x_real(x_real>255)  =255;x_real(x_real<0)=0;
x = double2q(x_real);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
Az=blurq(x,ker);
d1 = d1 +mu1*(x-z1) ;
d2 = d2 + mu2*(Az-z2);
mu1 = rho*mu1;
mu2 = rho*mu2;
%% Check the tolerence   
error=norm(x_real-q2double(x_prev),'fro')/norm(x_real,'fro'); 
disp(['error on step ' num2str(nstep)  ' is ' num2str(error) ', '...
    'and PSNR is ' num2str(psnr(x_real./255,x0/255)) ', and SSIM is ' num2str(ssim(x_real./255,x0./255)) ] );  

psnr_out = psnr(x_real./255,x0/255) ;  error_out = error ;
PSNR_out(nstep) = psnr_out ;     ERROR_out(nstep) = error_out ;

 if error<tol
        break;
 end
end
x=x_real;
figure(1999) ; grid on ;
subplot(1,3,1); plot(1:length(ERROR_out), ERROR_out) ;  grid on ;
subplot(1,3,2); plot(1:length(PSNR_out), PSNR_out) ;  grid on ;

function x = blurq(x,blur)
x1=x.w;
xi=x.x;
xj=x.y;
xk=x.z;
x1=imfilter(x1,blur,'circular');
xi=imfilter(xi,blur,'circular');
xj=imfilter(xj,blur,'circular');
xk=imfilter(xk,blur,'circular');
x=quaternion(x1, xi,xj, xk);