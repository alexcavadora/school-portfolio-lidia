%Calcular la distancia mínima del punto p(3,2) a la curva y = x^2
clear all, close all;
dx = 1e-4;
%f = @(x1, x2, x3) 6*x1*x2 + x2*x2 + x1*x3 + x3*x3;
f = @(x1, x2, x3) sqrt((x1-3).^2+(x2-2).^2) + x3 * (x2-x1.^2);

df1 =  @(x1, x2, x3) (f(x1 + dx, x2, x3) - f(x1 - dx, x2, x3))/(2*dx);
df2 =  @(x1, x2, x3) (f(x1, x2 + dx, x3) - f(x1, x2 - dx, x3))/(2*dx);
df3 =  @(x1, x2, x3) (f(x1, x2, x3 + dx) - f(x1, x2, x3 - dx))/(2*dx);
Gf = @(x1, x2, x3) [
					df1(x1, x2, x3);
					df2(x1, x2, x3);
					df3(x1, x2, x3)
				   ];
H11 = @(x1, x2, x3) (df1(x1 + dx, x2, x3) - df1(x1 - dx, x2, x3))/(2*dx);
H12 = @(x1, x2, x3) (df1(x1, x2 + dx, x3) - df1(x1, x2 - dx, x3))/(2*dx);
H13 = @(x1, x2, x3) (df1(x1, x2, x3 + dx) - df1(x1, x2, x3 - dx))/(2*dx);
H22 = @(x1, x2, x3) (df2(x1, x2 + dx, x3) - df2(x1 , x2 - dx, x3))/(2*dx);
H23 = @(x1, x2, x3) (df2(x1, x2, x3 + dx) - df2(x1, x2, x3 - dx))/(2*dx);
H33 = @(x1, x2, x3) (df3(x1, x2, x3 + dx) - df3(x1, x2, x3 + dx))/(2*dx);

H = @(x1, x2, x3) [
					H11(x1, x2, x3) H12(x1, x2, x3) H13(x1, x2, x3);
					H12(x1, x2, x3) H22(x1, x2, x3) H23(x1, x2, x3);
					H13(x1, x2, x3) H23(x1, x2, x3) H33(x1, x2, x3)
				  ];

x = [0.1; 0.1; 0.1];
err = 1e10;
while (err > 1e-5)
	%x2 = x - H(x(1), x(2), x(3)) \ Gf(x(1), x(2), x(3))
	x2 = x - inv(H(x(1), x(2), x(3))) * Gf(x(1), x(2), x(3));
	err = norm(x2-x);
	x = x2;
end
x
x(1)
x(2)
g = @(x) x.^2;
t = -2:0.1:3;
pbaspect ("manual")
plot (t, g(t), 'r', 3, 2,'*b', x(1), x(2), '*g'); hold on
plot([3 x(1)], [2 x(2)], 'black'); hold off	
%target = sqrt((x-3)^2 + (f(x)-2)^2) + lambda * (y - ^2)