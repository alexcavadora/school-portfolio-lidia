clear all;
close all;

X = sym('x', [1, 2]);

beale_func = (1.5 - X(1) + X(1) * X(2))^2 + ...
             (2.25 - X(1) + X(1) * X(2)^2)^2 + ...
             (2.625 - X(1) + X(1) * X(2)^3)^2;

r = @(x) double(subs(beale_func, X, x));
J = @(x) double(subs(gradient(beale_func, X), X, x));

X0 = [3, 0.5];
D = length(X0);
lambda = 0.01;
err = inf;
n = 1;
max_iter = 250;

while (err > 1e-5 && n < max_iter)
    J_X0 = J(X0);
    r_X0 = r(X0);
    delta = inv(J_X0' * J_X0 + lambda * eye(D)) * J_X0 * r_X0';

    X = X0 + delta';
    n = n + 1;
    err = norm(X - X0);
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
grid on;
saveas(gcf, 'lm_beale.png');
X0
n
err;
