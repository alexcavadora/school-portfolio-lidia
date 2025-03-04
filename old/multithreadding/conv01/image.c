#include "image.h"

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
   return image;
}

int image_delete( image_t* image ) {
   if ( image == NULL ) return 0;
   if ( image->pixmap == NULL ) return 0;
   free(image->pixmap);
   free(image);
   return 1;
}

image_t* image_load( const char* filename ) {
   unsigned width   = 0;
   unsigned height  = 0;
   unsigned error   = 0;
   unsigned char* data = NULL;;

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

image_t* image_conv( const image_t* image, double* matrix, unsigned r ) {
   image_t* conv = image_create(image->width, image->height);
   int W = image->width;
   int H = image->height;

   for ( int y=0; y<H; y++ ) {
      for ( int x=0; x<W; x++ ) {

         double sum = 0.0;
         for ( int m=-r; m<=r; m++ ) {
            for ( int n=-r; n<=r; n++ ) {

               // (2*r+1)*(m+r)+(n+r)

               sum += image->pixmap[W*(y+m)+(x+n)] * \
                      matrix[(2*r+1)*(m+r)+(n+r)];

            }
         }

      }
   }
}

