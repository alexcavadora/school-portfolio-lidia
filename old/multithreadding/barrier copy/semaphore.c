#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

//global
pthread_mutex_t g_bmut = PTHREAD_MUTEX_INITIALIZER;
unsigned        g_nthreads = 0;
unsigned        g_count = 0;
sem_t           g_sem_barrier;

void check_wake()
{
    pthread_mutex_lock(&g_bmut);
    g_count++;
    if (g_count == g_nthreads)
    {    g_count = 0;
        pthread_mutex_unlock(&g_bmut);
        for (int i = 0; i < g_nthreads; i++)
            sem_post(&g_sem_barrier);
    }
    else
    {
        pthread_mutex_unlock(&g_bmut);
        sem_wait(&g_sem_barrier);
    }
}

void* thread_function(void* arg)
{
    int id = *((int*) arg);
    free(arg);
    printf("Primera etapa, hilo %i\n", id);
    check_wake();
    printf("Seinda etapa, hilo %i\n", id);
    check_wake();
    printf("Tercera etapa, hilo %i\n", id);
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

    sem_init(&g_sem_barrier, 0, 0);

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
