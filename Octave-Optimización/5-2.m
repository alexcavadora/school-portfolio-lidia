clear all, close all;
pkg load optim;
% Define the objective function
objective_function = @(x) x(1).^2 - 5 * x(2);

% Initial point for optimization
initial_point = [1; 6];

% Linear inequality constraints
constraint_matrix_A = [1, 1];
constraint_vector_b = [5];

% No linear equality constraints
equality_constraint_matrix_Aeq = [];
equality_constraint_vector_beq = [];

% Bounds for variables
lower_bounds = [-Inf, -Inf];
upper_bounds = [10, Inf];

% Start timer
tic;

% Perform constrained optimization
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

