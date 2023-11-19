function z = entropy(x)
% Compute entropy z=H(x) of a discrete variable x.
% Input:
%   x: a integer vectors  
% Output:
%   z: entropy z=H(x)
% Written by Mo Chen (sth4nth@gmail.com).
n = numel(x);
[u,~,x] = unique(x);
k = numel(u);
idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
Px = nonzeros(mean(Mx,1));
Hx = -dot(Px,log2(Px));
z = max(0,Hx);
end

function z = jointEntropy(x, y)
% Compute joint entropy z=H(x,y) of two discrete variables x and y.
% Input:
%   x, y: two integer vector of the same length 
% Output:
%   z: joint entroy z=H(x,y)
% Written by Mo Chen (sth4nth@gmail.com).    
assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);
l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));
idx = 1:n;
p = nonzeros(sparse(idx,x,1,n,k,n)'*sparse(idx,y,1,n,k,n)/n); %joint distribution of x and y
z = -dot(p,log2(p));
z = max(0,z);
end

function z = condEntropy (x, y)
% Compute conditional entropy z=H(x|y) of two discrete variables x and y.
% Input:
%   x, y: two integer vector of the same length 
% Output:
%   z: conditional entropy z=H(x|y)
% Written by Mo Chen (sth4nth@gmail.com).
assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);
l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));
idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log2(Pxy));
Py = nonzeros(mean(My,1));
Hy = -dot(Py,log2(Py));
% conditional entropy H(x|y)
z = Hxy-Hy;
z = max(0,z);
end

function z = mutInfo(x, y)
% Compute mutual information I(x,y) of two discrete variables x and y.
% Input:
%   x, y: two integer vector of the same length 
% Output:
%   z: mutual information z=I(x,y)
% Written by Mo Chen (sth4nth@gmail.com).
assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);
l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));
idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log2(Pxy));
Px = nonzeros(mean(Mx,1));
Py = nonzeros(mean(My,1));
% entropy of Py and Px
Hx = -dot(Px,log2(Px));
Hy = -dot(Py,log2(Py));
% mutual information
z = Hx+Hy-Hxy;
z = max(0,z);
end

function z = nmi(x, y)
% Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
% Input:
%   x, y: two integer vector of the same length 
% Ouput:
%   z: normalized mutual information z=I(x,y)/sqrt(H(x)*H(y))
% Written by Mo Chen (sth4nth@gmail.com).
assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);
l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));
idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log2(Pxy));
% hacking, to elimative the 0log0 issue
Px = nonzeros(mean(Mx,1));
Py = nonzeros(mean(My,1));
% entropy of Py and Px
Hx = -dot(Px,log2(Px));
Hy = -dot(Py,log2(Py));
% mutual information
MI = Hx + Hy - Hxy;
% normalized mutual information
z = sqrt((MI/Hx)*(MI/Hy));
z = max(0,z);
end

function z = nvi(x, y)
% Compute normalized variation information z=(1-I(x,y)/H(x,y)) of two discrete variables x and y.
% Input:
%   x, y: two integer vector of the same length 
% Output:
%   z: normalized variation information z=(1-I(x,y)/H(x,y))
% Written by Mo Chen (sth4nth@gmail.com).
assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);
l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));
idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log2(Pxy));
Px = nonzeros(mean(Mx,1));
Py = nonzeros(mean(My,1));
% entropy of Py and Px
Hx = -dot(Px,log2(Px));
Hy = -dot(Py,log2(Py));
% nvi
z = 2-(Hx+Hy)/Hxy;
z = max(0,z);
end

function z = relatEntropy (x, y)
% Compute relative entropy (a.k.a KL divergence) z=KL(p(x)||p(y)) of two discrete variables x and y.
% Input:
%   x, y: two integer vector of the same length 
% Output:
%   z: relative entropy (a.k.a KL divergence) z=KL(p(x)||p(y))
% Written by Mo Chen (sth4nth@gmail.com).    
assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);
l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));
idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Px = nonzeros(mean(Mx,1));
Py = nonzeros(mean(My,1));
z = -dot(Px,log2(Py)-log2(Px));
z = max(0,z);
end


k = 10;  % variable range
n = 100;  % number of variables
x = ceil(k*rand(1,n));
y = ceil(k*rand(1,n));
% x = randi(k,1,n);  % need statistics toolbox
% y = randi(k,1,n);
%% Entropy H(x), H(y)
Hx = entropy(x);
Hy = entropy(y);
%% Joint entropy H(x,y)
Hxy = jointEntropy(x,y);
%% Conditional entropy H(x|y)
Hx_y = condEntropy(x,y);
%% Mutual information I(x,y)
Ixy = mutInfo(x,y);
%% Relative entropy (KL divergence) KL(p(x)|p(y))
Dxy = relatEntropy(x,y);
%% Normalized mutual information I_n(x,y)
nIxy = nmi(x,y);
%% Nomalized variation information I_v(x,y)
vIxy = nvi(x,y);
%% H(x|y) = H(x,y)-H(y)
isequalf(Hx_y,Hxy-Hy)
%% I(x,y) = H(x)-H(x|y)
isequalf(Ixy,Hx-Hx_y)
%% I(x,y) = H(x)+H(y)-H(x,y)
isequalf(Ixy,Hx+Hy-Hxy)
%% I_n(x,y) = I(x,y)/sqrt(H(x)*H(y))
isequalf(nIxy,Ixy/sqrt(Hx*Hy))
%% I_v(x,y) = (1-I(x,y)/H(x,y))
isequalf(vIxy,1-Ixy/Hxy)

