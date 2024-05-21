#include <stdio.h>
int main() {
	int n = 0;
	scanf("%d", &n);
	int array[n];
	int counter = 0;
	for (int i = 0; i < n; i++)
		scanf("%d", &array[i]);
	for(int i = 0; i < n; i++){
		counter = 0;
		for(int j = 0; j < n; j++){
			if(array[j] == array[i])
				counter++;
		}
    if(counter > 1) {
			printf("verdadero");
			return 0;
		}
	}
	printf("falso");
   return 0;
}