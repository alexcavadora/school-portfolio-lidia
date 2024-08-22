

%syms x y lambda;

%f = x * y;

%g = x + y - 10;

%L = f + lambda * g;

%dL_dx = diff(L, x);
%dL_dy = diff(L, y);
%dL_dlambda = diff(L, lambda);


%[sol_x, sol_y, sol_lambda] = solve(dL_dx == 0, dL_dy == 0, dL_dlambda == 0, x, y, lambda);

%sol_x
%sol_y


syms x y lamda k r;

f =( 2*x )*(2*y.^2)*k;

g = x.^2 + y.^2 - r.^2;

L = f + g*lamda;

dL_dx = diff(L,x);
dL_dy = diff(L,y);
dL_dl = diff(L,lamda);

[sol_x, sol_y, sol_l] = solve(dL_dx==0, dL_dy==0, dL_dl==0, x, y, lamda);

sol_x
sol_y

















