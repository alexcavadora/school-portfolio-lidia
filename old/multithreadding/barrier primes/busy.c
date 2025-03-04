#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// Cuenta los hilos que llegaron a la barrera
pthread_mutex_t g_bmut     = PTHREAD_MUTEX_INITIALIZER;
unsigned        g_count1   = 0;
unsigned        g_count2   = 0;
unsigned        g_nthreads = 0;

void* thread_func( void* arg ) {
   int id = *( (int*)arg );
   free(arg);

   // proceso
   printf("Primera Etapa %u\n",id);

   // barrera
   pthread_mutex_lock(&g_bmut);
   g_count1 ++;
   pthread_mutex_unlock(&g_bmut);
   while ( g_count1 < g_nthreads ); // Bussy waiting

   // proceso
   printf("Segunda Etapa %u\n",id);

   pthread_mutex_lock(&g_bmut);
   g_count2 ++;
   pthread_mutex_unlock(&g_bmut);
   while ( g_count2 < g_nthreads ); // Bussy waiting

   printf("Tercera Etapa %u\n",id);

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
   if ( p == NULL ) return NULL;
   *p = n;
   return p;
}

int main( int argc, char** argv ) {

   if ( argc != 2 ) print_help();

   g_nthreads = atoi(argv[1]);
   if ( g_nthreads == 0 || g_nthreads > 32 )
      error_msg("Wrong number of threads");

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

