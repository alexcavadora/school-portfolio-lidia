clear all, close all;
pkg load optim;

% Define the objective function
objective_function = @(x) x(1).^2 + 10 * x(2).^2 - 3 * x(1) * x(2);

% Initial point for optimization
initial_point = [1; 1];

% No linear inequality constraints (set A and b to empty matrices)
constraint_matrix_A = [];
constraint_vector_b = [];

% No linear equality constraints
equality_constraint_matrix_Aeq = [];
equality_constraint_vector_beq = [];

% No bounds for variables (removed all constraints and bounds)
lower_bounds = [];
upper_bounds = [];

% Start timer
tic;

% Perform unconstrained optimization
[optimal_variables, minimum_value, exit_flag, optimization_output] = fminunc(objective_function, initial_point);

% Number of iterations taken (not available for unconstrained optimization)
num_iterations = NaN;

% Display results
disp("Iterations:");
disp(num_iterations);
disp("Optimal Variables:");
disp(optimal_variables);
disp("Minimum Value:");
disp(minimum_value);

% Stop timer and display elapsed time
toc;

% Plotting contour and optimal point
x1 = linspace(-5, 5, 100);
x2 = linspace(-5, 5, 100);
[X1, X2] = meshgrid(x1, x2);
Z = X1.^2 + 10 * X2.^2 - 3 * X1 .* X2;
contour(X1, X2, Z, 50);
hold on;
plot(optimal_variables(1), optimal_variables(2), 'ro', 'MarkerSize', 10);
hold off;
xlabel('x1');
ylabel('x2');
title('Contour Plot with Optimal Point (Part c)');

