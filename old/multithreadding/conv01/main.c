#include <stdio.h>
#include "image.h"

#define RADIUS 100

double dist2( int x0, int y0, int x1, int y1 ) {
   return (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0);
}

int main() {

   //image_t* img = image_create(600,400);

   image_t* img = image_load("foto.png");

   rgba_t alpha = { 0, 0, 0, 0};

   for ( unsigned y=0; y<img->height; y++ ) {
      for ( unsigned x=0; x<img->width; x++ ) {
         if ( dist2(img->width/2,img->height/2,x,y) < RADIUS*RADIUS ) {
            img->pixmap[y*img->width+x] = alpha;
         }
      }
   }

   image_save(img,"test.png");
   image_delete(img);
   return 0;
}

