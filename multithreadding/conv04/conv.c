#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include "image.h"

#define NBUF 15
#define CONVR 5

// typedef struct{
//     char* filename;
//     image_t* image;

// } imgext_t;

//--------------------------------------------------------------
pthread_mutex_t g_fmut   = PTHREAD_MUTEX_INITIALIZER;
FILE*           g_imlist = NULL;
atomic_bool     g_fhalt = 0;
atomic_bool     g_cphalt = 0;
atomic_bool     g_ohalt = 0;

pthread_mutex_t g_imut   = PTHREAD_MUTEX_INITIALIZER;
unsigned        g_in     = 0;       // Número de elementos
image_t*        g_ibuff[NBUF];  // Buffer de entrada

pthread_mutex_t g_omut   = PTHREAD_MUTEX_INITIALIZER;
unsigned        g_on     = 0;       // Número de elementos
image_t*        g_obuff[NBUF];  // Buffer de salida

// Semáforos
sem_t g_sem_pa;
sem_t g_sem_ca;
sem_t g_sem_pb;
sem_t g_sem_cb;


//--------------------------------------------------------------
int clear_br(char* str)
{
    int i = 0;
    while(str[i] != '\n' && str[i] != '\r' && str[i] != 0) // '\r' is for overwriting lines
        i++;
    str[i] = 0;
    if (str[i] == 0) return 1;
    else
    return 0;
}
void* thread_image_load( void* args ) {
   char fname[BUFSIZ];
   image_t* img;
   while(!g_fhalt) {
      sem_wait(&g_sem_pa);

      // averiguar que archivo leer
      pthread_mutex_lock(&g_fmut);
      if ( fgets(fname,BUFSIZ,g_imlist) == NULL ) g_fhalt = 1;
      pthread_mutex_unlock(&g_fmut);
      clear_br(fname);
      printf("Now loading image: %s\n", fname);

      // cargamos la image en memoria
      img = image_load(fname);
      //printf("Now loading image: %s_conv.jpg\n", img->fname);
      if ( img == NULL ) exit(2);// continue; // archivo no existe

      // ingresar la imagen en el buffer de entrada
      pthread_mutex_lock(&g_imut);
      g_ibuff[g_in] = img;
      g_in ++;
      pthread_mutex_unlock(&g_imut);

      sem_post(&g_sem_ca);
   }

   return NULL;
}

void* thread_image_conv( void* args ) {
   image_t *img, *conv;
   while(!g_cphalt) {
      sem_wait(&g_sem_ca);
      pthread_mutex_lock(&g_imut);
      img = g_ibuff[g_in-1];
      g_in --;
      pthread_mutex_unlock(&g_imut);

      if(g_in == 0 && g_fhalt) g_cphalt = 1;

      sem_post(&g_sem_pa);

      conv = image_conv(img,CONVR);
      conv->fname = img->fname;
      img->fname = NULL;
      image_delete(img);

      sem_wait(&g_sem_pb);

      pthread_mutex_lock(&g_omut);
      g_obuff[g_on] = conv;
      g_on ++;
      pthread_mutex_unlock(&g_omut);
      sem_post(&g_sem_cb);
   }
   return NULL;
}

void* thread_save_image(void* args)
{
    char fname[BUFSIZ];
    image_t* img;
    while(!g_ohalt)
    {
        sem_wait(&g_sem_cb);

        pthread_mutex_lock(&g_omut);
            img = g_obuff[g_on - 1];
            g_on --;
        pthread_mutex_unlock(&g_omut);
        if (g_on == 0 && g_cphalt) g_ohalt  = 1;

        sem_post(&g_sem_pb);

        printf("Now saving image: %s_conv.jpg\n", img->fname);
        sprintf(fname, "%s_conv.jpg", img->fname);
        image_save(img, fname);
        image_delete(img);
    }
    return NULL;
}

// imgext_t* imgext_load(const char* filename)
// {
//     return NULL;
// }
// imgext_t_save (const char* filename, imgext_t* img)
// {
//     return NULL;
// }

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
    sem_init(&g_sem_ca,0,0);
    sem_init(&g_sem_pb, 0, NBUF);
    sem_init(&g_sem_cb,0,0);

    pthread_t *threads_pa = malloc(cores* sizeof(pthread_t));
    // Creamos los hilos
    for (int i = 0; i < cores; i++)
    {
        pthread_create(&threads_pa[i], NULL, thread_image_load,NULL);
    }

    pthread_t *threads_cp = malloc(cores* sizeof(pthread_t));
    for (int i = 0; i < cores; i++)
    {
        pthread_create(&threads_cp[i], NULL, thread_image_conv,NULL);
    }

    pthread_t *threads_cb = malloc(cores* sizeof(pthread_t));
    for (int i = 0; i < cores; i++)
    {
        pthread_create(&threads_cb[i], NULL, thread_save_image,NULL);
    }

    // join
    for (int i = 0; i < cores; i++)
    {
        pthread_join(threads_pa[i], NULL);
    }
    for (int i = 0; i < cores; i++)
    {
        pthread_join(threads_cp[i], NULL);
    }
    for (int i = 0; i < cores; i++)
    {
        pthread_join(threads_cb[i], NULL);
    }

    fclose(g_imlist);
    free(threads_pa);
    free(threads_cp);
    free(threads_cb);
    return 0;
}
