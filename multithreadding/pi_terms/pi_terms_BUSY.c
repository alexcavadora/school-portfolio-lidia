#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

// CR Variables
int cr_turn = 0;
u_int64_t cr_batch_count = 0;
double cr_pi = 0;
//CR variables end

// Global
int* g_label = NULL;
int g_tnum = 0;
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


void* pi_digits( void* arg)
{
    u_int64_t k;
    double sum = 0.0;
    const int label = *((int*) arg);
    int halt = 0;

    while(1)
    {
        //CR enter
        while(cr_turn == label) break;
        cr_pi += sum;

        if (cr_batch_count < g_batch_size)
        {
            k = cr_batch_count * g_n_batches;
            cr_batch_count++;
        }
        else halt = 1;
        //change turns to a still active thread
        int i = (label + 1 == g_tnum) ? 0 : label + 1 == g_tnum;
        while (g_label[i] < 0)
        {
            i = (i + 1) % g_tnum;
        }
        cr_turn = i;
        //CR exit
        if (halt) //make this thread unaccesable from the cycle above and finish the thread
        {
            g_label[label] = -1;
            return NULL;
        }

        sum = 0.0;
        for(u_int64_t i = 0; i < g_batch_size; i++)
        {
            sum += (1.0 - 2.0 * (k & 1)) / ((double) (2.0 * k + 1.0));
            k++;
        }
    }
    return NULL;
}
int main(int argc, char** argv)
{
    if (argc != 4) print_help();

    int g_tnum = atoi(argv[1]);
    g_n_batches = atoll(argv[2]);
    g_batch_size = atoll(argv[3]);

    if (g_tnum > 200 || g_tnum < 1) error_msg("Invalid number of threads.");
    if (g_n_batches == 0) error_msg("Batch size has to be at least 1.");
    if (g_batch_size == 0) error_msg("You have to generate at least 1 number.");

    pthread_t* tlist = malloc(g_tnum * sizeof(pthread_t));
    if (tlist == NULL) error_msg("Couldn't initiate threads, free some memory.");

    g_label = malloc(g_tnum * sizeof(int)); // indices for the thread
    if (g_label == NULL) error_msg("Couldn't initiate threads, free some memory.");

    for (int i = 0; i < g_tnum; i++)
        g_label[i] = i;

    for(int i = 0; i < g_tnum; i++)
        pthread_create(&tlist[i], NULL, pi_digits, &g_label[i]);

    for(int i = 0; i < g_tnum; i++)
        pthread_join(tlist[i], NULL);

    cr_pi *= 4.0;
    printf("pi ≈ %.25g\n", cr_pi);
    free(tlist);
    free(g_label);
}
