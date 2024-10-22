#include <stdio.h>
#include <stdlib.h>
#include "image.h"

void error_msg( const char* msg ) {
   printf("ERROR: %s\n",msg);
   exit(2);
}

int main() {

   image_t* image = image_load("foto.png");
   if ( image == NULL ) error_msg("Image not loaded.");

   image_t* conv = image_conv( image, 5 );
   if ( conv == NULL ) error_msg("Convolution error.");

   image_save(conv, "conv.png");

   image_delete(image);
   image_delete(conv);
   return 0;
}

