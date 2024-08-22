%   Newton Raphson para raices

syms x x1;

f = x^2-2;
df_x = diff(f,x);
err = inf;
n=0;
iter=100;
x1 = 1;
old_x = x1;

while(n<iter && err>0.00000001)
    x1 = x1 - (subs(f,x,x1)/subs(df_x,x,x1));
    n = n+1;
    err = abs(x1-old_x);
    old_x = x1;
end

disp(['La raiz es ', num2str(double(x1))]);
disp(['El error es de ', num2str(double(err))]);
disp(['Numero de iteraciones ', num2str(n)]);



