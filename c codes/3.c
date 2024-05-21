#include <stdio.h>
#include <stdlib.h>
int main() {
	int g1, g2, r1;
	scanf("%d %d %d", &g1, &g2, &r1);
	int da = abs(g1 - r1);
	int db = abs(g2 - r1);
	if(da == db)
		printf("raton C");
	else if(da > db)
		printf("gato B");
	else
		printf("gato A");

   return 0;
}