%gradiente con restriccion

syms x y l;

f =  sqrt((x-3)^2 + (y-2)^2);
g  = (y-x^2);

L = f + l*g;
G = gradient(L, [x,y,l]);
        
x1 = [0.5; 0.5; 0.1];
n = 0;
iter = 1000;
a = 1e-3;
err = inf;


while(n<iter && err>1e-6)
    x2 = x1 - a * double((subs(G, {x,y,l}, {x1(1),x1(2),x1(3)})));
    err = norm(x2-x1);
    x1=x2;
    n=n+1;
end

disp('Valor final de x:');
disp(x2);
disp('Error final:');
disp(err);
disp('Número total de iteraciones:');
disp(n);
