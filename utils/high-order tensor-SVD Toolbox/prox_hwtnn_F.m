function [X,htnn,tsvd_rank] = prox_hwtnn_F(Y,hfun_sg,gamma,lambda,rho)

%The proximal operator for the order-D tensor nuclear norm under Discrete Fourier Transform (DFT)
%
% The proximal operator of the tensor nuclear norm of a n_d way tensor
%
% min_X rho*Sum(w_{i}*sigma_{i})+0.5*||X-Y||_F^2
%
% hfun_sg   -    functions
% Y     -    n1*n2*n3*...*nd tensor
% X     -    n1*n2*n3*...*nd tensor
% wtnn   -   weighted tensor nuclear norm of X
% trank -    tensor tubal rank of X
% Written by  Zhihui Tu
% 2023-4-27


p = length(size(Y));
n = zeros(1,p);
for i = 1:p
    n(i) = size(Y,i);
end

X = zeros(n);
L = ones(1,p);
for i = 3:p
    Y = fft(Y,[],i);
    L(i) = L(i-1) * n(i);
end

htnn = 0;
tsvd_rank = 0;
        
[U,S,V] = svd(Y(:,:,1),'econ');
S = diag(S);
w = hfun_sg(S,gamma,lambda) ; 
threshold_value = w*rho ;
r = length(find(S>threshold_value));
if r>=1
    S = S(1:r)-threshold_value(1:r);
    X(:,:,1) = U(:,1:r)*diag(S)*V(:,1:r)';
    htnn = htnn+sum(S);
    tsvd_rank = max(tsvd_rank,r);
end

for j = 3 : p
    for i = L(j-1)+1 : L(j)
   %
        I = unfoldi(i,j,L);
        halfnj = floor(n(j)/2)+1;
   %
        if I(j) <= halfnj && I(j) >= 2
            [U,S,V] = svd(Y(:,:,i),'econ');
            S = diag(S);
            w = hfun_sg(S,gamma,lambda) ; 
            threshold_value = w*rho ;
            r = length(find(S>threshold_value));
            if r>=1
                S = S(1:r)-threshold_value(1:r);
                X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
                htnn = htnn+sum(S)*2;
                tsvd_rank = max(tsvd_rank,r);
            end
            
        %Conjugation property
        elseif I(j) > halfnj
            %
            n_ = nc(I,j,n);
            %
            i_ = foldi(n_,j,L);
            X(:,:,i) = conj( X(:,:,i_));
                
        end
    end
end

htnn = htnn/prod(n(3:end));

for i = p:-1:3
    X = (ifft(X,[],i));
end
X = real(X);






