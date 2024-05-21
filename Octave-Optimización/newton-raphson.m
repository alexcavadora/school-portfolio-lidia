close all;
clear all;
dx = 1e-6;
#f = @(x) x.^3 + (1/2)*x.^2 + 6*x + 1;
f = @(x) x - cos(x);

df = @(x) (f(x + dx)-f(x - dx))/(2*dx);
err = inf;

x0 = 0;
while (err>1e-8)
	x1 = x0 - f(x0)/df(x0);
	err = abs(x1 - x0);
	x0 = x1
end
err
x = -5:0.01:5;
plot(x,f(x),'r', x0,0,'*b');