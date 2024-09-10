clear all; close all;


sphere_func = @(x) sum(x.^2);
sphere_grad = @(x) 2 * x;

N = 50;
D = 2;

Rangos = [-5 5; -5 5];

X0 = zeros(N, D);
for i = 1:D
    a = Rangos(i, 1);
    b = Rangos(i, 2);
    for j = 1:N
        X0(j, i) = a + (b - a) * rand();
    end
end

f = sphere_func;
err = inf;
n = 0;
maxiter = 12000;
tol = 1e-12;
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
figure(1);
contourf(X1, X2, F, 10);
hold on;
plot(sphere_func(X0), '*r');
colorbar;
grid on;
[xo, ~] = min(X0);
err = f(xo);
xo
err
n

