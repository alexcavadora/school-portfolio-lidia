/*
Número palíndromo.
Dado un número entero X. Imprimir "verdadero" si X es un número palíndromo y "falso" en caso de no serlo
*/

#include <stdio.h>
int main()
{
	int size = 10;
	char n[size];
	scanf("%s", n);

	int n_size = 0;
	for(int i = 0; i < size; i++){
		if(n[i] == '\0')
			break;
		n_size = i;
	}

	int palindromos = 0;
	for(int i = 0; i < n_size; i++){
		if(n[i] == n[n_size-i])
			palindromos++;
		else
			break;
	}

	if(palindromos == n_size)
		printf("true");
	else
		printf("false");

	return 0;
}