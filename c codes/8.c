/*
Revertir un string
*/

#include <stdio.h>
int main()
{
	int size = 100;
	char str[size];
	scanf("%s", str);

	//find the length
	int length;
	for(length = 0; length < size; length++)
		if(str[length] == '\0')
			break;

	//calculate the reverse str and print as we go along
	char reversed_str[length];
	for(int i = 0; i < length; i++)
	{
		reversed_str[i] = str[length - i];
		printf("%c",reversed_str[i]);
	}
	return 0;

}