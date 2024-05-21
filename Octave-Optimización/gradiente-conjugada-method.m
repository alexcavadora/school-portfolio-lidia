clear all, close all;
A = [1 0 1;
	 0 2 1;
	 1 0 0];

B = [2; 5; 1];

% vector r (Gradiente)
x = [0.1; 0.1; 0.1];


r = A*x - B
				% inicializadores
Pk = B;		

err = inf;

while (err>1e-8)
	ak = (transpose(Pk)*r)/(transpose(Pk)*A*Pk);
	xi = x - ak * Pk;

	r = r - ak * (A * Pk)
	bk = -(transpose(Pk)* A *r)/(transpose(Pk)*A*Pk);
	Pk = r + bk * Pk;

	err = norm(xi - x);
	x = xi;
end

x