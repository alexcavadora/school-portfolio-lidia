clear all; close all;

beale_func = @(x) (1.5 - x(1) + x(1) * x(2))^2 + (2.25 - x(1) + x(1) * x(2)^2)^2 + (2.625 - x(1) + x(1) * x(2)^3)^2;

N = 50;
D = 2;
Rangos = [-4.5 4.5; -4.5 4.5];
X0 = zeros(N, D);

for i = 1:D
    a = Rangos(i, 1);
    b = Rangos(i, 2);
    for j = 1:N
        X0(j, i) = a + (b - a) * rand();
    end
end

f = beale_func;
err = inf;
n = 0;
maxiter = 1000;
tol = 1e-5;
F = 0.75;
CR = 0.25;

while (err > tol && n < maxiter)
    U = zeros(size(X0));
    for j = 1:N
        k = randperm(N, 3);
        U(j, :) = X0(k(1), :) + F * (X0(k(2), :) - X0(k(3), :));
    end
    V = X0;
    for j = 1:N
        j_rand = randi(D);
        for i = 1:D
            if rand() < CR || i == j_rand
                V(j, i) = U(j, i);
            end
        end
    end

    for j = 1:N
        if f(V(j, :)) < f(X0(j, :))
            X0(j, :) = V(j, :);
        end
    end
    err = norm(X0(1, :) - X0(10, :));

    n = n + 1;
end

n
[xo, ~] = min(X0)
err = f(xo)

x1 = linspace(-4.5, 4.5, 100);
x2 = linspace(-4.5, 4.5, 100);
[X1, X2] = meshgrid(x1, x2);
F = zeros(size(X1));
for i = 1:numel(X1)
    F(i) = beale_func([X1(i), X2(i)]);
end

figure;
contourf(X1, X2, F, 50);
colorbar;
hold on;
plot(xo(1), xo(2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
saveas(gcf, 'ed_beale.png');
grid on;


