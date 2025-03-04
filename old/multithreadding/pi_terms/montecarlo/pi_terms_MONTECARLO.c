#include "pcg_basic.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>

//CR
pthread_mutex_t cr_mutex = PTHREAD_MUTEX_INITIALIZER;
u_int64_t cr_n_inside = 0;

// Global
int g_tnum = 0;
u_int64_t g_n_terms = 0;

typedef struct
{
    double x;
    double y;
} point_t ;

point_t random_point(pcg32_random_t* rng, uint64_t min, uint64_t max){
    point_t p;
    uint64_t n = max-min;
    p.x = n * (pcg32_random_r(rng) / (double)UINT32_MAX);
    p.y = n * (pcg32_random_r(rng) / (double)UINT32_MAX);
    return p;
}

void* pi_calc(void* arg)
{
    int size = g_n_terms/g_tnum;

    pcg32_random_t rng;
    pcg32_srandom_r(&rng, time(NULL), (uint64_t) &rng);
    point_t p;
    u_int64_t n = 0;

    for (int i = 0; i < size; i++){
        p = random_point(&rng, 0, 1);
        if (((p.x - 0.5)*(p.x - 0.5)) + ((p.y - 0.5)*(p.y - 0.5)) < 0.25){
            n++;
        }
    }
    pthread_mutex_lock(&cr_mutex);
    cr_n_inside += n;
    pthread_mutex_unlock(&cr_mutex);
    return NULL;
}

void print_help()
{
    printf("usage: ./pi_terms <n_threads> <digits_of_precision>");
}

void error_msg(const char * msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(2);
}

int main(int argc, char** argv) {
    if (argc != 3) print_help();

    g_tnum = atoi(argv[1]);
    g_n_terms = atoll(argv[2]);

    if (g_tnum > 200 || g_tnum < 1) error_msg("Invalid number of threads.");
    if (g_n_terms == 0) error_msg("Batch size has to be at least 1.");


    pthread_t* tlist = malloc(g_tnum * sizeof(pthread_t));
    if (tlist == NULL) error_msg("Couldn't initiate threads, free some memory.");


    for(int i = 0; i < g_tnum; i++){
        pthread_create(&tlist[i], NULL, pi_calc, NULL);
    }

    for(int i = 0; i < g_tnum; i++){
        pthread_join(tlist[i], NULL);
    }

    printf("PI ≈ %g\n", 4.0*((double)cr_n_inside/(double)g_n_terms));

    free(tlist);
    return 0;
}
