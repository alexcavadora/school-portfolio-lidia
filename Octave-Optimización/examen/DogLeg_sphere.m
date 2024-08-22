pkg load symbolic;
clear all;
close all;

syms x1 x2;
X = [x1, x2];
sphere_func = x1^2 + x2^2;

f = @(x) double(subs(sphere_func, [x1, x2], x));
df = @(x) double(subs(gradient(sphere_func, [x1, x2]), [x1, x2], x));
Hf = @(x) double(subs(hessian(sphere_func, [x1, x2]), [x1, x2], x));

X0 = [3; 4];
D = length(X0);
Err = inf;
n = 1;
lambda = 1e-4;
d = 3;

while (Err > 1e-5 && n < 550)
    gradX0 = df(X0');
    HessianX0 = Hf(X0');

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
    Err = norm(delta_dl);
    X0 = X;
end

X0
n
Err

x1 = -5:0.1:5;
x2 = -5:0.1:5;
[X1, X2] = meshgrid(x1, x2);
F = X1.^2 + X2.^2;
figure(1);
contourf(X1, X2, F, 10);
hold on;
plot(X0(1), X0(2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
colorbar
saveas(gcf, 'dl_sphere.png');
