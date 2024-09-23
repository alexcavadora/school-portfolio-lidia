#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>
#include <pthread.h>

sem_t* g_sem = NULL;
void* thread_func(void* arg)
{
    int label = *((int*) arg);
    free(arg);
    sem_wait(&g_sem[label]);
    printf("Hello from %i thread.\n", label);
    sem_post(&g_sem[label + 1]);
    return NULL;
}

void print_help()
{
    printf("usage: semaphore <n_threads>");
}

void error_msg(const char * msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(2);
}

int* ip(int n){
    int* p = malloc(sizeof(int));
    *p = n;
    return p;
}

int main(int argc, char** argv)
{
    if (argc != 2) print_help();
    int tnum = atoi(argv[1]);
    if (tnum > 200 || tnum < 1) error_msg("Invalid number of threads.");

    g_sem = malloc(tnum* sizeof(sem_t));
    pthread_t* tlist = malloc(tnum * sizeof(pthread_t));
    if (tlist == NULL) error_msg("Couldn't initiate threads, free some memory.");

    for(int i = 0; i < tnum; i++){
        sem_init(&g_sem[i], 0, 0);
    }
    sem_post(&g_sem[0]);
    for(int i = 0; i < tnum; i++){
        pthread_create(&tlist[i], NULL, thread_func, ip(i));
    }

    for(int i = 0; i < tnum; i++){
        pthread_join(tlist[i], NULL);
    }

    free(tlist);
    free(g_sem);
}
