#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

//global
unsigned g_nthreads = 0;
pthread_mutex_t g_bmut = PTHREAD_MUTEX_INITIALIZER;
unsigned        g_count1 = 0; //cuenta los hilos que llegaron a la barrera
unsigned        g_count2 = 0; //cuenta los hilos que llegaron a la barrera
unsigned        g_count3 = 0; //cuenta los hilos que llegaron a la barrera


void* thread_function(void* arg)
{
    int id = *((int*) arg);
    free(arg);

    int i = 0;
    //primera etapa
    for (i = 0; i < g_count1; i++)
    {
        printf("A");
    }
    if (i == g_count1)
        printf("\n");

    pthread_mutex_lock(&g_bmut);
    g_count1++;
    pthread_mutex_unlock(&g_bmut);
    while(g_count1 < g_nthreads);

    //segunda etapa
    int j = 0;
    for (j = 0; j < g_count2; j++)
    {
        printf("B");
    }
    if (j == g_count2)
        printf("\n");

    pthread_mutex_lock(&g_bmut);
    g_count2++;
    pthread_mutex_unlock(&g_bmut);
    while(g_count2 < g_nthreads);

    //tercera etapa
    int k = 0;
    for (k = 0; k < g_count3; k++)
    {
        printf("C");
    }
    if (k == g_count3)
        printf("\n");

    pthread_mutex_lock(&g_bmut);
    g_count3++;
    pthread_mutex_unlock(&g_bmut);
    while(g_count3 < g_nthreads);

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
    if(g_nthreads <= 0 || g_nthreads > 64) error("Invalid number of threads", 2);

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
