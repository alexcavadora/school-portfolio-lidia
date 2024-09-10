#include <stdio.h>
int main() {
	int t1, h1, t2, h2;
	scanf("%d %d", &t1, &h1);
	scanf("%d %d", &t2, &h2);
	if(t1 > t2 && h1 > h2)
		printf("Hueso 1");
	else if(t2 > t1 && h2 > h1)
		printf("Hueso 2");
	else
		printf("Perrito confundido :(");

   return 0;
}