%%%%%%%%%%%%%%%%%% dogleg.m %%%%%%%%%%%%%%%%%%
%
% This script produces a figure illustrating Powell's dog-leg method.
%
% It is tested with GNU Octave 4.02 and MATLAB 9.0.0.341360 (R2016a) under
% Xubuntu 16.04. It is intended to be used under Linux. I have no idea
% what will happen if you attempt to run it in Windows or Mac (though
% I doubt your computer would explode).
%
% Octave and MATLAB behave differently with some distances/sizes, and
% hence the code differs a bit in the two cases. Moreover, Octave uses
% "print -depslatexstandalone" and pdflatex to process the LaTeX elements,
% which necessitates the presence of some LaTeX distribution in the
% computer, which was Tex Live 2016 during the test.
%
% If you are using Octave, do not get frightened by the ugly figure
% poped on your screen, where the LaTeX elements are not processed yet.
% Check instead dogleg.eps and dogleg.pdf, which are decent.
%
% Some of the distances/sizes may not be ideal for other versions of
% Octave or MATLAB. Yet it is not difficult to adapt them.
%
% Outputs:
% The script produces dogleg.eps and dogleg.pdf, which contain the
% figure. Besides, it has three numeric outputs, namely Delta, g, and B.
% Delta is the trust region redius, the trust region being
% ||delta|| \le Delta
% (notice that we use delta to denote the trust region step, following
% the historical notations in in Powell [1, 2], the papers that invented
% dog-leg method).
% g and B are the gradient and Hessian of the quadratic function to be
% minimized within the trust region.
%
% The script is motivated by Figure 4.4 of Nocedal and Wright [3].
%
% You can use this script under the GNU General Public License version 3.
%
% References:
% [1] M. J. D. Powell, "A hybrid method for nonlinear equations", in: P. Robinowitz, ed., Numerical Methods for Nonlinear Algebraic Equations, Gordon and Breach Science, London, 1970, pp. 87--114
% [2] M. J. D. Powell, "A Fortran Subroutine for Solving Systems of Nonlinear Algebraic Equations", in: P. Robinowitz, ed., Numerical Methods for Nonlinear Algebraic Equations, Gordon and Breach Science, London, 1970, pp. 115--161
% [3] J. Nocedal and S. Wright, Numerical Optimization, Springer Science & Business Media, 2006
%
% Author: ZHANG Zaikun (http://www.zzk.me)
% 21 July 2016, Hong Kong Polytechnic University
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Delta, g, B] = dogleg()

if (exist('OCTAVE_VERSION', 'builtin')) % We are running Octave
    OCTAVE = 1;
    graphics_toolkit('gnuplot');
    % Define the function to draw an arrow
    drawArrow = @(x,y) quiver(x(1),y(1),x(2)-x(1),y(2)-y(1), 'AutoScale', 'off', 'maxheadsize', 0.013, 'Color', 'k', 'LineStyle', '-', 'LineWidth', 0.2);
    drawArrow_dashed = @(x,y) quiver(x(1),y(1),x(2)-x(1),y(2)-y(1), 'AutoScale', 'off', 'maxheadsize', 0.02, 'Color', 'k', 'LineStyle', '-.', 'LineWidth', 0.06);
else % We are likely running MATLAB
    OCTAVE = 0;
    drawArrow = @(x,y) quiver(x(1),y(1),x(2)-x(1),y(2)-y(1), 'AutoScale', 'off', 'Color', 'k', 'LineStyle', '-', 'LineWidth', 0.2);
    drawArrow_dashed = @(x,y) quiver(x(1),y(1),x(2)-x(1),y(2)-y(1), 'AutoScale', 'off', 'Color', 'k', 'LineStyle', '-.', 'LineWidth', 0.06);
end

phi = (sqrt(5)-1)/2; % The golden ratio

close all;
h = figure();

% Draw the Cauchy point
r_cauchy = phi;
theta_cauchy = -(1-phi)*pi/2;
cauchy = r_cauchy*[cos(theta_cauchy); sin(theta_cauchy)];
plot(cauchy(1), cauchy(2), 'k.', 'MarkerSize', 16);
if (OCTAVE == 1) % We are running Octave
    text(cauchy(1)-(1-phi)*phi, cauchy(2), 'Cauchy');
else % We are likely running MATLAB
    text(cauchy(1)- 0.44*phi, cauchy(2), 'Cauchy');
end
hold on;

% Draw the Newton point
%r_newton = (2-phi)^2;
theta_newton = (1-phi)*pi/2.266;
newton = cauchy + [cos(theta_newton); sin(theta_newton)];
plot(newton(1), newton(2), 'k.', 'MarkerSize', 16);
text(newton(1)-0.1, newton(2)+0.12*phi, 'Newton');

% Connect the Cauchy point and the Newton one
plot([cauchy(1) newton(1)], [cauchy(2) newton(2)], 'LineStyle', '-', 'Color', 'k');

% Draw Powell's dog-leg point
powell = phi*cauchy + (1-phi)*newton;
plot(powell(1), powell(2), 'r.', 'MarkerSize', 20);
text(powell(1)+0.1*phi, powell(2)-0.02, 'Powell');

% Draw the trust region
Delta = norm(powell); % Determine the trust region redius so that powell is indeed the dog-leg point
theta = (0 : 0.05 : 2*pi+0.05);
x = Delta*cos(theta);
y = Delta*sin(theta);
plot(x,y, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 0.2);
text(-phi, 0, '$\|\delta\| \le \Delta$', 'Interpreter', 'latex');

% Draw the trust region center
text(-0.14*phi, 0, '$\mathbf{x}$', 'Interpreter', 'latex');

% Draw the trust region redius
theta_d = (phi)*pi;
d = Delta*[cos(theta_d), sin(theta_d)];
drawArrow_dashed([0 d(1)], [0 d(2)]);
text(d(1)*phi-(1-phi)^2*phi, d(2)*phi, '$\Delta$', 'Interpreter', 'latex');

% Find out the underlying B and g, so that -B^{-1}g = newton and -(g'g/g'Bg)g = cauchy = -phi*g
C = [cauchy(1)^2, cauchy(2)^2, 2*cauchy(1)*cauchy(2); newton(1), 0, newton(2); 0, newton(2), newton(1)];
b = [cauchy'*cauchy; cauchy];
b = C\b;
cauchy_stepsize = phi;
B = [b(1), b(3); b(3), b(2)]/cauchy_stepsize;
g = -cauchy/cauchy_stepsize;
% See if B is positive definite
% eigB = eig(B)

% Draw the minus gradient
drawArrow([0 -g(1)], [0 -g(2)]);
plot(0, 0, 'k.', 'MarkerSize', 16);
text(-g(1)+0.04, -g(2), '$-\mathbf{g}$', 'Interpreter', 'latex');

% Draw the trajectory of the optimal solution
lambda = 0;
trajectory = -B\g;
multiplier = lambda;
increment = 0.025;
while (lambda < 100)
    lambda = lambda + increment/norm(((B+lambda*eye(2,2))^2)\g);
    trajectory = [trajectory, -(B+lambda*eye(2,2))\g];
    multiplier = [multiplier, lambda];
end
plot(trajectory(1, :), trajectory(2, :), 'LineStyle', '-.', 'LineWidth', 0.2, 'Color', 'k');

% Draw the intersection of the trajectory with the trust region boundary
normt = zeros(size(trajectory, 2), 1);
for i = 1:size(trajectory, 2)
    normt(i) = norm(trajectory(:, i));
end
[~, ind] = min(abs(normt - Delta));
t = trajectory(:, ind);
t = t/norm(t)*Delta;
plot(t(1), t(2), 'k.', 'MarkerSize', 16);
text(t(1)-0.3*phi, t(2)+0.1*phi, '$\delta^*(\Delta)$', 'Interpreter', 'latex');

axis ('equal', 'off');
box('off');
end

