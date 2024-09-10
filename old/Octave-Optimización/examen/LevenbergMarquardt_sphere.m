clear all;
close all;

sphere_func = @(x) x(1)^2 + x(2)^2;
sphere_grad = @(x) [2*x(1); 2*x(2)];

X0 = [3; -3]';
D = size(X0, 1);
Err = inf;
n = 1;
max_iter = 250;


while (Err > 1e-5 && n < max_iter)
    residual = sphere_func(X0);
    jacobian = sphere_grad(X0);

    delta = -pinv(jacobian) * residual;

    X = X0 + delta;
    Err = norm(delta);
    X0 = X;
    n = n + 1;
end

x1 = -5:0.1:5;
x2 = -5:0.1:5;
[X1, X2] = meshgrid(x1, x2);
F = X1.^2 + X2.^2;

figure;
contourf(X1, X2, F, 10);
colorbar;
hold on;
plot(X0(1), X0(2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
grid on;
saveas(gcf, 'lm_sphere.png');
X0
n
Err;

