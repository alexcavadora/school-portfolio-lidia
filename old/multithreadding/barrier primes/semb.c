#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

unsigned        g_nthreads = 0;

pthread_mutex_t g_bmut  = PTHREAD_MUTEX_INITIALIZER;
unsigned        g_count = 0;
sem_t           g_barrier1;
sem_t           g_barrier2;

void* thread_func( void* arg ) {
   int id = *( (int*)arg );
   free(arg);

   printf("Primera etapa, hilo %i\n",id);

   // Barrera
   pthread_mutex_lock(&g_bmut);
   g_count ++;
   if ( g_count == g_nthreads ) {
      g_count = 0;
      pthread_mutex_unlock(&g_bmut);
      for ( unsigned i=0; i<g_nthreads-1; i++ ) sem_post(&g_barrier1);
   }
   else {
      pthread_mutex_unlock(&g_bmut);
      sem_wait(&g_barrier1);
   }

   printf("Segunda etapa, hilo %i\n",id);

   // Barrera
   pthread_mutex_lock(&g_bmut);
   g_count ++;
   if ( g_count == g_nthreads ) {
      g_count = 0;
      pthread_mutex_unlock(&g_bmut);
      for ( unsigned i=0; i<g_nthreads-1; i++ ) sem_post(&g_barrier2);
   }
   else {
      pthread_mutex_unlock(&g_bmut);
      sem_wait(&g_barrier2);
   }

   printf("Tercera etapa, hilo %i\n",id);

   return NULL;
}

void print_help() {
   printf("usage: busy <threads>\n");
   exit(1);
}

void error_msg( const char* msg ) {
   fprintf(stderr,"ERROR:%s\n",msg);
   exit(2);
}

int* intp( int n ) {
   int* p = malloc( sizeof(int) );
   if ( p != NULL ) *p = n;
   return p;
}

int main( int argc, char** argv ) {

   if ( argc != 2 ) print_help();

   g_nthreads = atoi(argv[1]);
   if ( g_nthreads == 0 ) error_msg("Wrong number of threads");

   pthread_t* tlist = malloc( g_nthreads * sizeof(pthread_t) );
   if ( tlist == NULL ) error_msg("Allocation failure.");

   sem_init(&g_barrier1, 0, 0);
   sem_init(&g_barrier2, 0, 0);

   for ( unsigned i=0; i<g_nthreads; i++ ) {
      pthread_create(
            &tlist[i],
            NULL,
            thread_func,
            intp(i)
            );
   }

   for ( unsigned i=0; i<g_nthreads; i++ ) {
      pthread_join(tlist[i], NULL);
   }

   free(tlist);
   return 0;
}

