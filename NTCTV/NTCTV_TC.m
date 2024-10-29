%----- Tensor Correlted Total Viariation based Tensor Completion -----%
function [X, Y, iter, jiluchgX] = NTCTV_TC(M, Omega, X0, opts,data) 
% Solve the p-order Tensor Completion via Noline Tensor Correlted Total Viariation(NTCTV) norm minimization by ADMM
% the transform in high-order TSVD uses DFT (default)
%
% min_{X \in R^{n1*n_2*...*n_d}} ||X||_ntctv s.t. P_Omega(X) = P_Omega(M)
%
% ---------------------------------------------
% Input:
%       M       -    any p-order observed tensor
%       opts    -    Structure value in Matlab. The fields are
%           opts.directions          -   considered local smoothness along certain directions 
%           opts.transform           -   the transform case of TSVD, DFT, DCT and other invertible linear transform 
%           opts.transform_matrices  -   the transform matrices of TSVD for generalized invertible linear transform           
%           opts.tol                 -   termination tolerance
%           opts.max_iter            -   maximum number of iterations
%           opts.mu                  -   stepsize for dual variable updating in ADMM
%           opts.max_mu              -   maximum stepsize
%           opts.rho                 -   rho>=1, ratio that is used to increase mu
%           opts.detail             -   0 or 1, show the update details or not 
%
% Output:
%       X      -    recovered order-p tensor
%       obj    -    objective function value
%       err    -    residual 
%       iter   -    number of iterations
%

% updata 2023 /5 /25 by Zhang yigong


%% default paremeters setting 
% acfun = inline('tanh(x)','x');

dim = size(M);
d   = ndims(M);

directions = 1:3; % The smoothness of first two spatial dimensions is considered by default
tol        = 1e-6; 
max_iter   = 4000;
rho        = 1e-3; % Penalty parameters
detail     = 1;

newton_insweep = 30;

if ~exist('opts', 'var')
    opts = [];
end  

if isfield(opts, 'alpha');              alpha              = opts.alpha;              end
if isfield(opts, 'beta');               beta               = opts.beta;               end
if isfield(opts, 'r');                  r                  = opts.r;                end
if isfield(opts, 'directions');         directions         = opts.directions;         end
if isfield(opts, 'tol');                tol                = opts.tol;                end
if isfield(opts, 'max_iter');           max_iter           = opts.max_iter;           end
if isfield(opts, 'rho');                rho                = opts.rho;                end
if isfield(opts, 'newton_insweep');     newton_insweep     = opts.newton_insweep;     end
if isfield(opts, 'detail');             detail             = opts.detail;             end

%% variables initialization
n        = length(directions);
X        = X0;
X(Omega) = M(Omega);
E        = M-X;

for i = 1:n
    index          = directions(i);
    G{index}       = porder_diff(X0,index); 
    temp           = Unfold(G{index},size(G{index}),3);
    [D{index},~,~] = svds(temp,r);
%     temp           = dftmtx(dim(3));
%     D{index}       = temp(1:r,:)';
    Z{index}       = Fold(D{index}' * Unfold(G{index},size(G{index}),3),[dim(1),dim(2),r],3);
    Y{index}       = tanh_my(Z{index});
end

%% FFT setting
T = zeros(dim);
for i = 1:n
    Eny = diff_element(dim,directions(i));
    T   = T + Eny; 
end

%% main loop
iter = 0;
while iter<max_iter
    iter = iter + 1;  
    Xk = X;
    Ek = E;
    %% Update X -- solve TV by FFT 
    H = zeros(dim);
    for i = 1:n
       index = directions(i);
       H = H + porder_diff_T(alpha*G{index},index); 
    end
    X = real( ifftn( fftn( beta*(M-E)+rho*Xk+H)./((beta+rho)+alpha*T) ) );
    %% Updata Gi -- close form solution
    for i = 1:n
        index = directions(i);
        temp = Fold(D{index}*Unfold(Z{index},size(Z{index}),3),dim,3);
        G{index} = (alpha*(porder_diff(X,index)+temp)+rho*G{index})./(2*alpha+rho);
    end
    
    %% Updata Yi -- proximal operator of TNN
    for i = 1:n
        index = directions(i);
        temp = (alpha * tanh_my(Z{index}) + rho * Y{index}) ./ (alpha + rho);
        Y{index} = my_SVD(temp,1/(n*(alpha+rho)));
    end
    %% update Zi ¡ª¡ª use newton methon
    for i =1:n
        index      = directions(i);
        Z_mat      = Unfold(Z{index},size(Z{index}),3);
        temp1      = Unfold(Y{index},size(Y{index}),3);
        DG         = Fold(D{index}' * Unfold(G{index},size(G{index}),3),[dim(1),dim(2),r],3);
        temp2      = (alpha * DG + rho * Z{index}) / (alpha + rho);
        temp2      = Unfold(temp2,size(temp2),3);
        Z_mat      = Newton(temp1,temp2,Z_mat,newton_insweep,alpha + rho,alpha);
        Z{index}   = Fold(Z_mat,size(Z{index}),3);
    end
    
    %% update Di
    for i =1:n
        index = directions(i);
        Z_mat   =  Unfold(Z{index},size(Z{index}),3);
        G_mat   =  Unfold(G{index},size(G{index}),3);
        temp =  alpha * G_mat * Z_mat' + rho * D{index};
        [UI,~,VI] = svd( temp , 'econ');
        D{index}   = UI * VI';  
    end
    
    %% Update E 
    E          = (beta*(M-X)+rho*Ek)./(beta+rho);
    E(Omega)   = 0;
    
    %% Stop criterion
%     dY   = M-X-E;    
    %chgX = max(abs(Xk(:)-X(:)));
    chgX = norm(Xk(:)-X(:))/norm(Xk(:));
    %chgE = max(abs(Ek(:)-E(:)));
    %chg  = max([chgX chgE max(abs(dY(:)))]);
    if iter>10
        if chgX < tol
            break;
        end
    end
    jiluchgX(iter) = chgX; 
    %% Update detail display
    if detail
        if iter == 1 || mod(iter, 200) == 0
            ppssnr = PSNR3D(255*data,255*X);
%             err = norm(dY(:),'fro');
%             disp(['iter ' num2str(iter) ', err=' num2str(err)  ', chgX=' num2str(chgX) ', chgX=' num2str(ppssnr)]);
            disp(['iter ' num2str(iter) ', chgX=' num2str(chgX) ', psnr=' num2str(ppssnr)]);
        end
    end
    
end
end

%% nested functions
    function Z  = Newton(g,a,Z,inner,alpha, beta)
            i=0;
            relchg=1;
            tol=10^(-4);  
            while  i < inner  &&  relchg > tol 
                    Zp=Z;
                    Numer = beta .* (1 - tanh_my(Z).^2) .* (tanh_my(Z) - g) + alpha .* (Z - a);
                    Denom = -2 .* beta .* tanh_my(Z) .* ( 1 - tanh_my(Z).^2) .* (tanh_my(Z) - g) + beta .* ( 1 - tanh_my(Z).^2).^2 + alpha;
                    Z  = Z - Numer./Denom;
                    relchg = norm(Z - Zp,'fro')/norm(Z,'fro');
                    i=i+1;
            end
    end

    function Y = my_SVD(Y,rho)
    [n1, n2, n3] = size(Y);
    n12 = min(n1, n2);
    Uf = zeros(n1, n12, n3);
    Vf = zeros(n2, n12, n3);
    Sf = zeros(n12,n12, n3);
    trank = 0;
    for i = 1 : n3
        [Uf(:,:,i), Sf(:,:,i), Vf(:,:,i)] = svd(Y(:,:,i), 'econ');
        s = diag(Sf(:, :, i));
        s = max(s - rho, 0);
        Sf(:, :, i) = diag(s);
        temp = length(find(s>0));
        trank = max(temp, trank);
        Y(:,:,i) = Uf(:,:,i)*Sf(:,:,i)*Vf(:,:,i)';
    end
    end
% 
   function output = tanh_my(x)
      output =(exp(x) - exp(-x))./(exp(x) + exp(-x));
   end
