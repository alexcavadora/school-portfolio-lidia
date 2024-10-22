#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include "image.h"

#define NBUF 15
#define CONVR 5

//--------------------------------------------------------------
pthread_mutex_t g_fmut   = PTHREAD_MUTEX_INITIALIZER;
FILE*           g_imlist = NULL;

pthread_mutex_t g_imut   = PTHREAD_MUTEX_INITIALIZER;
unsigned        g_in     = 0;       // Número de elementos
image_t*        g_ibuff[NBUF];  // Buffer de entrada

pthread_mutex_t g_omut   = PTHREAD_MUTEX_INITIALIZER;
unsigned        g_on     = 0;       // Número de elementos
image_t*        g_obuff[NBUF];  // Buffer de salida

// Semáforos
sem_t g_sem_pa;
sem_t g_sem_cp;
sem_t g_sem_cb;

//--------------------------------------------------------------

void* thread_image_load( void* args ) {
   char fname[BUFSIZ];
   image_t* img;
   int halt = 0; // indica que el hilo terminó

   while( 1 ) {
      sem_wait(&g_sem_pa);

      // averiguar que archivo leer
      pthread_mutex_lock(&g_fmut);
      if ( fgets(fname,BUFSIZ,g_imlist) == NULL ) halt = 1;
      pthread_mutex_unlock(&g_fmut);

      if ( halt ) return NULL;

      // cargamos la image en memoria
      img = image_load(fname);
      if ( img == NULL ) continue; // archivo no existe

      // ingresar la imagen en el buffer de entrada
      pthread_mutex_lock(&g_imut);
      g_ibuff[g_in] = img;
      g_in ++;
      pthread_mutex_unlock(&g_imut);

      sem_post(&g_sem_ca);
      //sem_post(&g_sem_pa); // Checar!!
   }

   return NULL;
}

void* thread_image_conv( void* args ) {
   image_t* img, conv;

   while( 1 ) {
      sem_wait(&g_sem_ca);

      pthread_mutex_lock(&g_imut);
      img = g_ibuff[g_in-1];
      g_in --;
      pthread_mutex_unlock(&g_imut);

      sem_post(&g_sem_pa);

      conv = image_conv(img,CONVR);
      image_delete(img);

      pthread_mutex_lock(&g_omut);
      g_obuff[g_on] = conv;
      g_on ++;
      pthread_mutex_unlock(&g_omut);

      sem_post(&g_sem_cb);
   }

   return NULL;
}


//--------------------------------------------------------------
void print_help() {
   printf("usage: conv <cores> <list.txt>\n");
   printf("list.txt   Lista de archivos a procesar.\n");
   printf("cores      Número de núcleos del sistema.\n");
   exit(1);
}

void error_msg( const char* msg ) {
   fprintf(stderr,"ERROR:%s\n",msg);
   exit(2);
}

int main( int argc, char** argv ) {

   if ( argc != 3 ) print_help();

   unsigned cores = atoi(argv[1]);
   char* filename = argv[2];

   if ( cores < 1 || cores > 32 ) error_msg("Wrong number of cores.");
   g_imlist = fopen(filename,"rt");
   if ( g_imlist == NULL ) error_msg("File not found.");

   // Inicializamos los semáforos
   sem_init(&g_sem_pa,0,NBUF);
   sem_init(&g_sem_cp,0,0);
   sem_init(&g_sem_cb,0,0);

   // Creamos los hilos
   // join

   return 0;
}

