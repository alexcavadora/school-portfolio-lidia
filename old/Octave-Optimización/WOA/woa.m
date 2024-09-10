% Paso 0: Carga de datos e hiperparámetros
pkg load symbolic;
clear all;
close all;

% Cargar los datos
data = csvread('DatosAgua.csv');
t = data(:, 1);
Y = data(:, 2);

g = @(x) x(1) * exp(x(2) * t) + x(3);

% Parámetros de WOA
num_whales = 50;
max_iter = 25000;
dim = 3;
lb = [-100, -1, 20];
ub = [100, 1, 50];
a = 2;
convergence_threshold = 1e-25;
convergence_patience = 10000;

% Paso 1: Inicializar la población de ballenas
pos = zeros(num_whales, dim);
for i = 1:dim
    pos(:, i) = lb(i) + (ub(i) - lb(i)) * rand(num_whales, 1);
end
best_pos = pos(1, :);

% Paso 2: Calcular la aptitud inicial
best_score = inf;
for i = 1:num_whales
    current_pos = pos(i, :);
    g_vals = g(current_pos');
    mse = mean((Y - g_vals).^2);
    if mse < best_score
        best_score = mse;
        best_pos = current_pos;
    end
end

% Paso 3: Bucle de iteraciones
t_iter = 0;
no_improvement_counter = 0;

while t_iter < max_iter
    a = 2 - t_iter * (2 / max_iter);

    for i = 1:num_whales
        r1 = rand();
        r2 = rand();
        A = 2 * a * r1 - a;
        C = 2 * r2;

        p = rand();
        if p < 0.5
            if abs(A) < 1
                D = abs(C * best_pos - pos(i, :));
                pos(i, :) = best_pos - A * D;
            else
                rand_pos = pos(randi([1 num_whales]), :);
                D = abs(C * rand_pos - pos(i, :));
                pos(i, :) = rand_pos - A * D;
            end
        else
            D_prime = abs(best_pos - pos(i, :));
            l = -1 + (1 + 1) * rand();
            pos(i, :) = D_prime .* exp(1) * cos(2 * pi * l) + best_pos;
        end

        for d = 1:dim
            pos(i, d) = max(min(pos(i, d), ub(d)), lb(d));
        end

        current_pos = pos(i, :);
        g_vals = g(current_pos');
        mse = mean((Y - g_vals).^2);
        if mse < best_score
            best_score = mse;
            best_pos = current_pos;
            no_improvement_counter = 0;
        else
            no_improvement_counter = no_improvement_counter + 1;
        end
    end

    t_iter = t_iter + 1;


    if mod(t_iter, 100) == 0
        disp(['Iteration: ' num2str(t_iter) ' Best score: ' num2str(best_score)]);
    end
end

% Paso 4: Impresión de resultados
g_best = g(best_pos);
figure;
plot(t, g_best, 'g');
hold on;
plot(t, Y, 'b');
legend('Modelo Ajustado', 'Datos Originales');
xlabel('Tiempo');
ylabel('Temperatura');
title('Optimización de la Temperatura del Agua usando WOA');
ylim([20 35]);

disp('Best parameters:');
disp(best_pos);
disp('Best score:');
disp(best_score);
rmse = sqrt(best_score);
disp(['RMSE: ', num2str(rmse)]);

