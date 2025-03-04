#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

//global
pthread_mutex_t g_cmut = PTHREAD_MUTEX_INITIALIZER;
unsigned        g_nthreads = 0;
unsigned        g_count = 0;
pthread_cond_t  g_condv = PTHREAD_COND_INITIALIZER;

void check_wake()
{
    pthread_mutex_lock(&g_cmut);
    g_count++;
    if (g_count == g_nthreads)
    {
        g_count = 0;
        pthread_cond_broadcast(&g_condv);
    }
    else
    {
        pthread_cond_wait(&g_condv, &g_cmut);
        // al despertar, bloquea otra vez y continua la ejecución
    }
    pthread_mutex_unlock(&g_cmut);
}

void* thread_function(void* arg)
{
    int id = *((int*) arg);
    free(arg);
    int stage = 0;
    while (stage < 4) {
        printf("Stage %i, thread %i\n", stage, id);
        check_wake();
        stage++;
    }
    return NULL;
}

void error(char* help, int r)
{
    printf("%s", help);
    exit(r);
}

int * intp(int n)
{
    int*p = malloc(sizeof(int));
    if(p==NULL) return p;
    *p = n;
    return p;
}

int main(int argc, char** argv){
    if (argc != 2) error("Usage: busy <n_threads>", 1);
    g_nthreads = atoi(argv[1]);
    if(g_nthreads <= 0) error("Wrong number of threads", 1);
    pthread_t* tlist = malloc(g_nthreads * sizeof(pthread_t));
    if (tlist == NULL) error("Couldn't allocate thread memory", 2);
    for(unsigned i = 0; i < g_nthreads; i++)
    {
        pthread_create(&tlist[i], NULL, thread_function, intp(i));
    }

    for(unsigned i = 0; i < g_nthreads; i++)
    {
        pthread_join(tlist[i], NULL);
    }
    free(tlist);
}
