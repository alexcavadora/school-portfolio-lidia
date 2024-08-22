clear all, close all
pkg load symbolic;
Data = csvread('DatosAgua.csv');

t = Data(:,1);
Y = Data(:,2);

figure(1), hold on
plot(t, Y, '.r')

f_model = @(x) x(1)*(1 - exp(-x(2)*t))+x(3);
abs_error = @(x) Y - f_model(x);
f = @(x) 0.5 * (abs_error(x)'*abs_error(x));


N_var = 3; % Numero de variables
N_pop = 50; % Número de la población
lower_bound = 1;
upper_bound = 10;

error = inf;
max_iterations = 1e4;

positions = lower_bound + (upper_bound - lower_bound) * rand(N_pop, N_var);
fitness = zeros(N_pop, 1);

chasers_pos = zeros(N_pop,N_var);
chasers_score = ones(N_pop,1) * 1e8;

wingers_pos = zeros(N_pop,N_var);
wingers_score = ones(N_pop,1) * 1e8;

cheaters_pos = zeros(N_pop,N_var);
cheaters_score = ones(N_pop,1) * 1e8;


iteration = 0;
while ((error > 1e-3) && (iteration < max_iterations))
    % Calculamos la aptitud de cada León
    for i = 1:N_pop
        fitness(i,:) = f(positions(i,:));
    end
    fitness(fitness == Inf) = max(fitness(isfinite(fitness)));

    % Actualizamos las posiciones de chasers, wingers y cheaters
    chasers_mask = fitness < chasers_score;
    chasers_score = fitness .* chasers_mask + chasers_score .* ~chasers_mask;
    chasers_pos = positions .* chasers_mask + chasers_pos .* ~chasers_mask;
    %chasers_score(isnan(chasers_score)) = max(chasers_score);

    wingers_mask = (fitness > chasers_score) & (fitness < wingers_score);
    wingers_score = fitness .* wingers_mask + wingers_score .* ~wingers_mask;
    wingers_pos = positions .* wingers_mask + wingers_pos .* ~wingers_mask;
    %wingers_score(isnan(wingers_score)) = max(wingers_score);

    cheaters_mask = (fitness > chasers_score) & (fitness > wingers_score) & (fitness < cheaters_score);
    cheaters_score = fitness .* cheaters_mask + cheaters_score .* ~cheaters_mask;
    cheaters_pos = positions .* cheaters_mask + cheaters_pos .* ~cheaters_mask;
    %cheaters_score(isnan(cheaters_score)) = max(cheaters_score);

    % Actualizamos la posición de los Leones
    r1 = rand(N_pop, N_var);
    r2 = rand(N_pop, N_var);
    lion_movement_1 = 2 * r1 - 1;
    lion_movement_2 = 2 * r2;
    distance_from_cheaters = abs(lion_movement_2 .* cheaters_pos - positions);
    position_change_1 = cheaters_pos - lion_movement_1 .* distance_from_cheaters;
    position_change_2 = chasers_pos - lion_movement_1 .* distance_from_cheaters;
    position_change_3 = (chasers_pos + wingers_pos + cheaters_pos) / 3.0;
    positions = (position_change_1 + position_change_2 + position_change_3) / 3.0;

    [~,index] = sort(chasers_score);
    error = f(chasers_pos(index(1), :))
    vpa(chasers_pos(index(1), :),10)% norm(chasers_pos(index(1), :) - chasers_pos(index(2), :));
    iteration = iteration + 1;
end

iteration
error
chasers_pos(index(1), :)

plot(t, f_model(chasers_pos(index(1), :)), 'b')

hold off
