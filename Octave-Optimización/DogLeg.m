pkg load symbolic;
clear all;
close all;

sphere_func = @(x) x(1)^2 + x(2)^2;
sphere_grad = @(x) [2*x(1); 2*x(2)];

X0 = [3; 4];
D = size(X0, 1);
Err = inf;
n = 1;
lambda = 1e-4;
d = 3;


while (Err > 1e-5 && n < 550)
    gradX0 = sphere_grad(X0);
    HessianX0 = 2 * eye(D);

    A = HessianX0 + lambda * eye(D);
    B = gradX0;

    delta_gn = -inv(A) * B;
    delta_sd = -gradX0;
    alpha = dot(delta_sd, delta_sd) / dot(HessianX0 * delta_sd, HessianX0 * delta_sd);

    if norm(delta_gn) <= d
        delta_dl = delta_gn;
    elseif alpha * norm(delta_sd) >= d
        delta_dl = (d / norm(delta_sd)) * delta_sd;
    else
        a = alpha * delta_sd;
        b = delta_gn;
        c = a' * (b - a);
        if c <= 0
            beta = (-c + sqrt(c^2 + dot(b - a, b - a) * (d^2 - dot(a, a)))) / dot(b - a, b - a);
        else
            beta = (d^2 - dot(a, a)) / (c + sqrt(c^2 + dot(b - a, b - a) * (d^2 - dot(a, a))));
        end
        delta_dl = alpha * delta_sd + beta * (delta_gn - alpha * delta_sd);
    end

    X = X0 + delta_dl;
    n = n + 1;
    Err = norm(X - X0);
    X0 = X;
end

x1 = -5:0.1:5;
x2 = -5:0.1:5;
[X1, X2] = meshgrid(x1, x2);
F = X1.^2 + X2.^2;
figure(1);
contourf(X1, X2, F, 10);
hold on;

colorbar;
grid on;

X0
n
Err

