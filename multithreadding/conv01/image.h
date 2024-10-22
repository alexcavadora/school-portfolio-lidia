#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>
#include <stdlib.h>
#include "lodepng.h"

typedef struct {
   uint8_t r;
   uint8_t g;
   uint8_t b;
   uint8_t a;
} rgba_t; // pixel

typedef struct {
   unsigned width;
   unsigned height;
   unsigned size;
   rgba_t*  pixmap;
} image_t;

image_t* image_create( unsigned width, unsigned height );
int image_delete( image_t* image );

// Carga una imagen
image_t* image_load( const char* filename );
int image_save( const image_t* image, const char* filename );

image_t* image_conv( const image_t* image, double* matrix, unsigned n );

#endif /* IMAGE_H */

