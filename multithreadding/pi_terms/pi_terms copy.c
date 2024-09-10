#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>


// CR Variables
pthread_mutex_t cr_mutex = PTHREAD_MUTEX_INITIALIZER;
int cr_turn = 0;
u_int64_t cr_batch_count = 0;
double cr_pi = 0.0;
//CR variables end

// Global
int g_tnum = 0;
u_int64_t g_n_terms = 0;

void print_help()
{
    printf("usage: pi_terms <n_threads> <batch_size> <batch_count>");
}

void error_msg(const char * msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(2);
}


void* pi_digits( void* arg){
    double sum = 0.0;
    int label = *((int*) arg);
    free(arg);
    int size = g_n_terms/g_tnum;
    u_int64_t k = size * label;
    printf("%llu\n", k);
    sum = 0.0;
    for(u_int64_t i = 0; i < size; i++)
    {
        sum += (double)(1.0 - 2.0 * (k & 1)) / (2.0 * k + 1.0);
        k++;
    }
    pthread_mutex_lock(&cr_mutex);
    cr_pi += sum;
    pthread_mutex_unlock(&cr_mutex);
    return NULL;
}

int* ip(int n){
    int* p = malloc(sizeof(int));
    *p = n;
    return p;
}


int main(int argc, char** argv)
{
    if (argc != 3) print_help();

    g_tnum = atoi(argv[1]);
    g_n_terms = atoll(argv[2]);

    if (g_tnum > 200 || g_tnum < 1) error_msg("Invalid number of threads.");
    if (g_n_terms == 0) error_msg("Batch size has to be at least 1.");

    pthread_t* tlist = malloc(g_tnum * sizeof(pthread_t));
    if (tlist == NULL) error_msg("Couldn't initiate threads, free some memory.");

    for(int i = 0; i < g_tnum; i++){
        pthread_create(&tlist[i], NULL, pi_digits, ip(i));
    }

    for(int i = 0; i < g_tnum; i++){
        pthread_join(tlist[i], NULL);
    }

    cr_pi *= 4.0;
    printf("pi ≈ %.25g\n", cr_pi);
    free(tlist);
}
