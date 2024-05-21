function manual1
  clear all, close all;
  # obtaining random data for the curve
  x = (0 : 0.5 : 10);
  f = @(t) -0.5 * exp(-0.5*t) + 1*t.*exp(-0.7*t);
  figure(1)
  plot(x, f(x)), hold on;
  a = 0.9;
  b = 1.1;
  x2 = x.*[a+(b-a)*rand(size(x))];
  y = f(x2).*[a+(b-a)*rand(size(x))];
  #figure(2)
  plot(x, y,'*')
  hold off;

  data = [x2, y];
  #clear x, a, b, x2, y;
  #close all

  t = data(:, 1);
  Y = data(:, 2);
  M = @(X) X(1) * exp(X(2)*t) + X(3)*exp(X(4)*t);
  F = @(X) (1/2) * transpose(Y-M(X))*[Y-M(X)];

  #F([1; 2; 0; 4])
  #F([-0.5; -0.5; 1.0; -0.7])

  y2 = GradF(F, [1;1;1;1])
end

function dy = GradF(F, X)
  dx = 1*e-6;
  n = size(X,1);
  Dx = eye(n)*dx;
  dy = zeros(size(X));
  for i= 1:n
    dy(i) = (F(X+Dx(:, i)) - F(X-Dx(:, i)))/ (2*dx);
  endfor
 endfunction

 function H = Hess(F,x)

 endfunction

