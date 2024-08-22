syms x y;


y =  x^2;
f = sqrt((x-3)^2 + (y-2).^2);
Gf = diff(f,x);
n= 0;
err = inf;
x1=0;
old = x1;
a = 0.01;
iter = 1000;

while(n<iter && err>1e-6)
    x1 = x1 - a*double(subs(Gf,x,x1));
    n=n+1;
    err = abs(x1-old);
    old = x1;
    
end    


x1
n
err


% si no converge modificar tasa de aprendizaje, las iteraciones y hacer
% doble la sustitucion del grdiente en x1