#include <stdio.h>
int main() {
	for (int i = 1; i < 1001; i++){
		if(i % 3 != 0 && i%5 != 0){
			printf("%i\n",i);
			continue;
		}
		if(i % 3 == 0)
			printf("Fizz");
		if(i % 5 == 0)
			printf("Buzz");
		printf("\n");
	}
   return 0;
}