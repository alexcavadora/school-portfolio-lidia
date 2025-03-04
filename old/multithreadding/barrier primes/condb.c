#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

unsigned        g_nthreads = 0;

pthread_mutex_t g_cmut  = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  g_condv = PTHREAD_COND_INITIALIZER;
unsigned        g_count = 0;

void* thread_func( void* arg ) {
   int id = *( (int*)arg );
   free(arg);

   int stage = 0;
   while ( stage < 5 ) {

      printf("Etapa %i, hilo %i\n",stage,id);

      // Barrera
      pthread_mutex_lock(&g_cmut);
      g_count ++;
      if ( g_count == g_nthreads ) {
         g_count = 0;
         pthread_cond_broadcast(&g_condv);
      }
      else {
         pthread_cond_wait(&g_condv, &g_cmut);
      }
      pthread_mutex_unlock(&g_cmut);

      stage ++;
   }

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

