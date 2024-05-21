#PCA requiere datos centrados para funcionar, y así es como rota
clear all, close all;
a = 3, b = 1;
t = 0:0.01:2*pi;
x = a*cos(t);
y = b*sin(t);
theta = 45*(pi/180);
R = [cos(theta), - sin(theta); sin(theta), cos(theta)];
X = R * [x;y];
x = X(1,:);
y = X(2,:);
figure(1)
plot(x,y);
X2 = X-mean(X,2);
N = size(X2,2);

C1 = cov(X2');
#[u,d] = eig(C1) # deprecated fct
[u,D,V]= svd(C1');

#test
#f = @(x) x - cos(x); #ddf = @(x) (df(x+dx)-df(x-dx))/dx; #ddf = @(x) (df(x+dx)-df(x-dx))/dx;
X3 = V * X2;

figure(2)
plot(X3(1,:),X3(2,:));