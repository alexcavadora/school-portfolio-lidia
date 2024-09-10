D = [1 2; 6 1; 3 3; 4 5];
[n, dim] = size(D);
m = 2;
X = D(:,1);
Y = D(:,2);

Z = ones(m+1,n);
for i = 2 : m+1
	Z(i,:) =(X.^(i-1))';
end

W = inv(Z*Z')*Z*Y;

plot(X, Y, 'b'); hold on

X2 = (min(X)-1: 0.1: max(X) + 1)';
Z2 = ones(m+1, size(X2,1));

for i = 2 : m+1
	Z2(i,:) =(X2.^(i-1))';
end
Y2 = Z2'*W;
plot(X2,Y2,'r', 'd');
