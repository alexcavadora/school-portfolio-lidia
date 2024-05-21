clear all, close all;
pkg load optim;

% Define the objective function
objective_function = @(x) x(1).^2 + 10 * x(2).^2 - 3 * x(1) * x(2);

% Initial point for optimization
initial_point = [1; 0.5];  % Adjust as needed

% Linear inequality constraints
constraint_matrix_A = [2, 1; 1, 1];
constraint_vector_b = [4; 5];

% No linear equality constraints
equality_constraint_matrix_Aeq = [];
equality_constraint_vector_beq = [];

% Bounds for variables
lower_bounds = [-5, -5];
upper_bounds = [5, 5];

% Start timer
tic;

% Perform constrained optimization with adjusted options
[optimal_variables, minimum_value, exit_flag, optimization_output] = fmincon(objective_function, initial_point, constraint_matrix_A, ...
    constraint_vector_b, equality_constraint_matrix_Aeq, equality_constraint_vector_beq, lower_bounds, upper_bounds);

% Number of iterations taken
num_iterations = optimization_output.niter;

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
title('Contour Plot with Optimal Point (Part a)');

