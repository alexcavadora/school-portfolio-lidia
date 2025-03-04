#include <stdio.h>

int main()
{
    printf("int = %lu bits\n", 8* sizeof(int));
    printf("long int = %lu bits\n", 8* sizeof(long int));
    printf("long long int = %lu bits\n", 8* sizeof(long long int));
}
