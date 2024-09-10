pkg load symbolic;
clear all;
close all;

sphere_func = @(x) x(1)^2 + x(2)^2;
sphere_grad = @(x) [2*x(1); 2*x(2)];

x1 = -5:0.1:5;
x2 = -5:0.1:5;
[X1, X2] = meshgrid(x1, x2);
F = X1.^2 + X2.^2;


X0 = [3; 4];
D = size(X0, 1);
Err = inf;
n = 1;
lambda = 1;
max_iter = 250;


while (Err > 1e-15 && n < max_iter)
    gradX0 = sphere_grad(X0);
    HessianX0 = 2 * eye(D);

    A = HessianX0 + lambda * eye(D);
    B = gradX0;

    delta = -inv(A) * B;
    X = X0 + delta;

    Err = norm(delta);


    if sphere_func(X) < sphere_func(X0)
        lambda = lambda / 10;
    else
        lambda = lambda * 10;
    end

    X0 = X;
    n = n + 1;
end


figure(1);
contourf(X1, X2, F, 10);
hold on;
plot(sphere_func(X0), '*r');
colorbar;
grid on;

disp('Optimized parameters:');
disp(X0);
disp('Number of iterations:');
disp(n);
disp('Final error:');
disp(Err);

