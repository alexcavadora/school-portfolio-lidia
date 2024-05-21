#include <stdio.h>
int main() {
	int length = 10; 
	char cadena[length];
	scanf("%s", cadena);
	int parentesis = 0, bracket = 0, curly_bracket = 0;
	for (int i = 0; i < lenght; i++){
		if(cadena[i] == '(')
			parentesis = 1;
		if(cadena [i] == ')')
			parentesis++;
		
		if(cadena[i] == '[')
			bracket = 1;
		if(cadena [i] == ']')
			bracket++;

		if(cadena[i] == '{')
			curly_bracket = 1;
		if(cadena [i] == '}')
			curly_bracket++;
		
		if(cadena[i] == '\0')
			break;
	}

	if((parentesis == 2 || parentesis == 0) && (bracket == 2 || bracket == 0) && (curly_bracket == 2 || curly_bracket == 0))
		printf("verdadero")
	else
		printf("falso");


}