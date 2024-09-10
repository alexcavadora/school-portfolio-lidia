close all; clear all;
dx = 1e-6;
f = @(x) x.^3 + 1;

df = @(x) (f(x + dx)-f(x - dx))/(2*dx);


ddf = @(x) (f(x + dx)- 2*f(x)+(f(x)-dx))/(dx.^2);
err = inf;

x0 = 1;
while (err>1e-6)
	x1 = x0 - df(x0)/ddf(x0);
	err = abs(x1 - x0);
	x0 = x1
	iter = iter + 1; 
	if (iter > 100) break;
end
err
x = -100:0.01:100;
plot(x,f(x),'r', x0,f(x0),'b');
plot(x0,f(x0),'*g');