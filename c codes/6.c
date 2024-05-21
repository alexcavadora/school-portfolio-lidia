#include <stdio.h>
bool check(char[] test);

int main() {
	int size = 10;
	char cadena[size];
	scanf("%s", cadena);
	int parentesis = 0, bracket = 0, curly_bracket = 0;
	for (int i = 0; i < size; i++){
		if(cadena[i] == '(')
			parentesis++;
		if(cadena[i] == ')'  && parentesis != 0)
			parentesis--;
		
		if(cadena[i] == '[')
			bracket++;
		if(cadena[i] == ']'  && bracket != 0)
			bracket--;

		if(cadena[i] == '{')
			curly_bracket++;
		if(cadena[i] == '}' && curly_bracket != 0)
			curly_bracket--;
		
		if(cadena[i] == '\0')
			break;
	}

	if(parentesis == 0 &&  bracket == 0 && curly_bracket == 0)
		printf("verdadero");
	else
		printf("falso");
}
bool check(char[] test){

}