pkg load symbolic;
clear all;
close all;

X = sym('x', [1, 2]);
beale_func = (1.5 - X(1) + X(1) * X(2))^2 + ...
             (2.25 - X(1) + X(1) * X(2)^2)^2 + ...
             (2.625 - X(1) + X(1) * X(2)^3)^2;
f = @(x) double(subs(beale_func, X, x));
df = @(x) double(subs(gradient(beale_func, X), X, x));
Hf = @(x) double(subs(hessian(beale_func, X), X, x));

X0 = [3; -3];
D = size(X0, 1);
Err = inf;
n = 1;
lambda = 1;
d = 3;
cof = 1;

while (Err > 1e-5 && n < 500)
    gradX0 = df(X0');
    HessianX0 = Hf(X0');
    A = HessianX0 + lambda * eye(D);
    B = -gradX0(:);
    delta = linsolve(double(A), double(B));
    t = (norm(B) / norm(HessianX0 * B)) ^ 2;


    if (norm(delta) <= cof)
        hdl = delta;
    elseif (norm(B * t) >= cof)
        hdl = (cof / norm(B)) * B;
    else
        a = t * B;
        b = delta;
        c = a' * (b - a);
        r = sqrt(c^2 + norm(b - a)^2 * (cof^2 - norm(a)^2));
        if (c <= 0)
            beta = (-c + r) / norm(b - a)^2;
        else
            beta = (cof^2 - norm(a)^2) / (c + r);
        end
        hdl = t * B + beta * (delta - t * B);
    end
    X = X0 + hdl;
    n = n + 1;
    Err = norm(delta);
    X0 = X;
end

x1 = -4.5:0.1:4.5;
x2 = -4.5:0.1:4.5;
[X1, X2] = meshgrid(x1, x2);
F = (1.5 - X1 + X1.*X2).^2 + (2.25 - X1 + X1.*X2.^2).^2 + (2.625 - X1 + X1.*X2.^3).^2;
figure(1);
contourf(X1, X2, F, 50);
hold on;
plot(X0(1), X0(2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');

colorbar;
grid on;
saveas(gcf, 'dl_beale.png');
X0
n
Err

