#include "pcg_basic.h"
#include <stdio.h>
#include <time.h>
#include <math.h>

int main() {
    pcg32_srandom(time(NULL), time(NULL));
    uint32_t rngX, rngY;
    double random_number, random_number2;
    long int N = 0, K = 200000000;
    for (int i = 0; i < K; i++){
        rngX = pcg32_random();
        random_number = (double)rngX/(double)UINT32_MAX;
        rngY = pcg32_random();
        random_number2 = (double)rngY/(double)UINT32_MAX;

        if (sqrt(pow(random_number - 0.5, 2.0) + pow(random_number2 - 0.5, 2.0)) <= 0.5){
            //printf("Inside: [%f, %f]\n", random_number, random_number2);
            N++;
        }
    }

    printf("Pi aprox = %f\n", 4.0*((double)N/(double)K));
    return 0;
}
