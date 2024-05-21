clear all; close all;
pkg load symbolic
X = sym('x', [2, 1]);

fs = X(1)^2*X(2)+X(2)^3*X(1)

f = @(x) subs(fs, X, x)
f([2,2])

df = @(x) subs(gradient(fs, X ), X, x);
df([1, 2])

Hf = @(x) subs(hessian(fs,X), X, x);
Hf([1, 2])
