#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

//CR Variables
u_int64_t cr_count = 0;
double cr_pi = 0;
//CR variables end

//global
u_int64_t g_n_batches = 0;
u_int64_t g_batch_size = 0;

void print_help()
{
    printf("usage: pi_terms <n_threads> <batch_size> <batch_count>");
}

void error_msg(const char * msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(2);
}


void* pi_digits( void* argv){
    u_int64_t k;
    double sum = 0.0;
    while(1){
        //CR enter
        if (cr_count >= g_n_batches * g_batch_size) return NULL;
        cr_pi += sum;
        k = cr_count;
        cr_count += g_batch_size;
        //CR exit
        sum = 0.0;
        for(u_int64_t i = 0; i < g_batch_size; i++){
            sum += (1.0 - 2.0 *(k & 1)) / ((double) (2.0 * k + 1.0));
            k++;
        }
    }
    return NULL;
}
int main(int argc, char** argv)
{
    if (argc != 4) print_help();

    int ntds = atoi(argv[1]);
    g_n_batches = atoll(argv[2]);
    g_batch_size = atoll(argv[3]);

    if (ntds > 200 || ntds < 1) error_msg("Invalid number of threads.");
    if (g_n_batches == 0) error_msg("Batch size has to be at least 1.");
    if (g_batch_size == 0) error_msg("You have to generate at least 1 number.");

    pthread_t* tlist = malloc(ntds * sizeof(pthread_t));
    if (tlist == NULL) error_msg("Couldn't initiate threads, free some memory.");

    for(int i = 0; i < ntds; i++){
        pthread_create(&tlist[i], NULL, pi_digits, NULL);
    }

    for(int i = 0; i < ntds; i++){
        pthread_join(tlist[i], NULL);
    }

    cr_pi *= 4.0;
    printf("pi ~= %.25g\n", cr_pi);
}
