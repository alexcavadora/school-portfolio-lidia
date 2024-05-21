clear all, close all;
pkg load optim;

% Define the objective function
objective_function = @(x) x(1).^2 + 10 * x(2).^2 - 3 * x(1) * x(2);

% Initial point for optimization (feasible initial point)
initial_point = [1; 1];

% Linear inequality constraints
constraint_matrix_A = [];
constraint_vector_b = [];

% No linear equality constraints
equality_constraint_matrix_Aeq = [];
equality_constraint_vector_beq = [];

% Bounds for variables
lower_bounds = [-5, -5];
upper_bounds = [5, 5];

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

