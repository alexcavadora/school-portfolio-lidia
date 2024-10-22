#include "image.h"
#include <stdio.h>
#include <stdlib.h>

image_t* image_create( unsigned width, unsigned height ) {
   image_t* image = malloc( sizeof(image_t) );
   if ( image == NULL ) return NULL;

   rgba_t* pixmap = malloc( width * height * sizeof(rgba_t) );
   if ( pixmap == NULL ) {
      free(image);
      return NULL;
   }

   image->width  = width;
   image->height = height;
   image->size   = width * height;
   image->pixmap = pixmap;
   image->fname = NULL;
   return image;
}

int image_delete( image_t* image ) {
   if ( image == NULL ) return 0;
   if ( image->pixmap == NULL ) return 0;
   if ( image->fname != NULL ) free(image->fname);
   free(image->pixmap);
   free(image);
   return 1;
}

//creates a dynamic string for filename, without extension
char* get_file_name(const char* filename)
{
    int n = 0;
    while(filename[n]!= 0) n++;
    int i = n-1;

    while(filename[i] != '.' && i > 0) i--;

    if(i < 0) i = n+1;

    char* fname = malloc(i * sizeof(char));
    if(fname == NULL) return NULL;

    //copy characters before i, file name before extension
    for (int k = 0; k<i; k++) fname[k] = filename[k];
    fname[i] = 0;
    return fname;
}
image_t* image_load( const char* filename ) {
   unsigned width   = 0;
   unsigned height  = 0;
   unsigned error   = 0;
   unsigned char* data = NULL;

   error = lodepng_decode32_file(&data, &width, &height, filename);
   if ( error ) return 0;

   image_t* image = malloc( sizeof(image_t) );
   if ( image == NULL ) {
      free(data);
      return NULL;
   }

   image->width  = width;
   image->height = height;
   image->size   = width * height;
   image->pixmap = (rgba_t*)data;
   image->fname = get_file_name(filename);
   return image;
}

int image_save( const image_t* image, const char* filename ) {
   unsigned error = 0;

   error = lodepng_encode32_file(
         filename,
         (unsigned char*)image->pixmap,
         image->width,
         image->height
         );

   if ( error ) return 0;
   return 1;
}

image_t* image_conv( const image_t* image, int r ) {
   if ( r <= 0 ) return NULL;
   int W = image->width;
   int H = image->height;
   int K = 2*r + 1;        // Lado de la matriz en pixeles
   image_t* conv = image_create(W,H);

   for ( int y=0; y<H; y++ ) {
      for ( int x=0; x<W; x++ ) {

         int sum_r = 0.0;
         int sum_g = 0.0;
         int sum_b = 0.0;

         for ( int m=-r; m<=r; m++ ) {
            int my = m + y;
            if ( my < 0 || my >= H ) continue;

            for ( int n=-r; n<=r; n++ ) {
               int nx = n + x;
               if ( nx < 0 || nx >= W ) continue;

               sum_r += image->pixmap[W*(my)+(nx)].r;
               sum_g += image->pixmap[W*(my)+(nx)].g;
               sum_b += image->pixmap[W*(my)+(nx)].b;
            }
         }

         // pixeles en la región sumada.
         conv->pixmap[W*y+x].r = (uint8_t)(sum_r/(K*K));
         conv->pixmap[W*y+x].g = (uint8_t)(sum_g/(K*K));
         conv->pixmap[W*y+x].b = (uint8_t)(sum_b/(K*K));
         conv->pixmap[W*y+x].a = 255;
      }
   }

   return conv;
}
