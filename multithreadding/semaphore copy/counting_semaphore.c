#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>
#include <pthread.h>

void print_help()
{
    printf("usage: semaphore <nthreads> <batch> <nprimes>\n");
    printf("nthreads:  Threads amount number.\n");
    printf("batch:      Amount of numbers to try per batch.\n");
    printf("nprimes:    Total amount of primes.\n");
}

void error_msg(const char * msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(2);
}

//critical region
pthread_mutex_t     cr_mutex = PTHREAD_MUTEX_INITIALIZER;
long long           cr_prime_count = 0;     // numbers in prime_list
long long*          cr_prime_list = NULL;   // prime number list
long long           cr_next = 2;            // next number to evaluate

//global variables
unsigned            g_nthreads = 0;
unsigned            g_batch = 0;
long long           g_nprimes = 0;

//semaphore list
sem_t* g_semlist = NULL;

int is_prime(long long num)
{
    if(num == 0) return 0;
    if(num == 1) return 0;
    if(num == 2) return 1;
    if(num == 3) return 1;
    if(num == 4) return 0;
    if(num == 5) return 1;

    if(num % 2 == 0) return 0;
    if(num % 5 == 0) return 0;

    long long nmax = 1 + 1 / (num * (1/2));
}
void* prime_thread(void* arg)
{
    int id = *((int*) arg);
    free(arg);
    long long int* list = malloc(sizeof(long long int) * g_batch/2);
    int cont = 1;
    long long num = 0;            // numero a probar
    long long end = 0;  // límite del batch
    unsigned n = 0;
    while (cont)
    {
        //--------------Critical_Region_start
        pthread_mutex_lock(&cr_mutex);
        if (n > 0)
        {
            for (unsigned i = 0; i < n; i++)
            {
                if (cr_prime_count == g_nprimes) cont = 0;
                cr_prime_list[cr_prime_count] = list[i];
                cr_prime_count++;
            }
        }
        if (cr_prime_count == g_nprimes) cont = 0;
        else
        {
            num = cr_next;
            cr_next += g_batch;
            end = cr_next;
        }
        pthread_mutex_lock(&cr_mutex);
        //----------------Critical_Region_end
       n = 0;
       while (num < end)
       {
          if (is_prime(num))
          {
             list[n] = num;
             n ++;
          }
          num ++;
       }
    }
    return NULL;
}

int* intp(int n){
    int* p = malloc(sizeof(int));
    if (p!=NULL)*p = n;
    return p;
}

int main(int argc, char** argv)
{
    if (argc != 4) print_help();
    g_nthreads = atoi(argv[1]);
    g_batch = atoi(argv[2]);
    g_nprimes =  atol(argv[3]);

    if (g_nthreads > 200 || g_nthreads < 1) error_msg("Invalid number of threads.");
    if (g_batch == 0) error_msg("Invalid number of batches.");
    if (g_nprimes == 0) error_msg("Invalid number of primes.");

    cr_prime_list = malloc(g_nprimes* sizeof(long long));
    if (cr_prime_list == NULL) error_msg("Couldn't initiate prime list, free some memory.");

    g_semlist = malloc(g_nthreads * sizeof(sem_t));
    if (g_semlist == NULL) error_msg("Couldn't initiate semaphores, free some memory.");
    sem_init(&g_semlist[0], 0, 1);
    for(unsigned i = 1; i < g_nthreads; i++){
        sem_init(&g_semlist[i], 0, 0);
    }

    pthread_t* tlist = malloc(g_nthreads * sizeof(pthread_t));
    if (tlist == NULL) error_msg("Couldn't initiate threads, free some memory.");

    for(int i = 0; i < g_nthreads; i++){
        pthread_create(&tlist[i], NULL, prime_thread, intp(i));
    }

    for(int i = 0; i < g_nthreads; i++){
        pthread_join(tlist[i], NULL);
    }

    free(tlist);
    free(g_semlist);
}
