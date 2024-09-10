#include "pcg_basic.h"
#include <stdio.h>
#include <stdint.h>
#include <time.h>

typedef struct
{
    double x;
    double y;
} point_t ;

uint64_t square(uint64_t n)
{
    if (n == 0)
        return 0;

    uint64_t x = n >> 1;

    if (n & 1)
        return ((square(x) << 2) + (x << 2) + 1);
    else
        return (square(x) << 2);
}
point_t random_point(pcg32_random_t* rng, uint64_t min, uint64_t max){
    point_t p;
    uint64_t n = max-min;
    p.x = n * (pcg32_random_r(rng) / (double)UINT32_MAX);
    p.y = n * (pcg32_random_r(rng) / (double)UINT32_MAX);
    return p;
}

int main() {
    pcg32_random_t rng;
    pcg32_srandom_r(&rng, time(NULL), (uint64_t) &rng);

    point_t p;
    long int n = 0, k = 20000000000;

    for (int i = 0; i < k; i++){
        p = random_point(&rng, 0, 1);
        if ((p.x - 0.5)*(p.x - 0.5) + (p.y - 0.5)*(p.y - 0.5) < 0.25){
            n++;
        }
    }

    printf("PI ≈ %g\n", 4.0*((double)n/(double)k));
    return 0;
}
