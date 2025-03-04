#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chess.h"


#ifndef MOVLEN
#define MOVLEN 128
#endif

#ifndef BUFLEN
#define BUFLEN 512
#endif

void trim( char* str ) {
   int i = 0;
   while ( str[i] != 0 ) {
      if ( str[i] == '\n' || str[i] == '\r' ) {
         str[i] = 0;
         break;
      }
      i++;
   }
}

int streq( const char* str1, const char* str2 ) {
   if ( strcmp(str1,str2) == 0 ) return 1;
   return 0;
}

int parse( const char* str, const char* lcom, const char* scom ) {
   if ( strcmp(str,lcom) == 0 ) return 1;
   if ( strcmp(str,scom) == 0 ) return 1;
   return 0;
}

int print_moves( int* moves, int sz ) {
  char str[3];
  if ( sz == 0 ) return 0;
  coord(moves[0],str);
  printf("%s",str);
  for ( int i=1; i<sz; i++ ) {
    coord(moves[i],str);
    printf(" %s",str);
  }
  printf("\n");
  return 1;
}

// 0 = No error, valid movement.
// 1 = There is no piece to move at position `a`.
// 2 = Wrong turn.
// 3 = Wrong move.
int print_move_error( int err ) {
  switch ( err ) {
    case 0:
      printf("No error.\n");
      break;
    case 1:
      printf("No piece to move.\n");
      break;
    case 2:
      printf("Wrong turn.\n");
      break;
    case 3:
      printf("Wrong move.\n");
      break;
  }
  return 0;
}

int main() {
   char buffer[BUFLEN];
   int  moves[MOVLEN];
   int  sz, pos, a, b, e;

   board_t* bd = board_create();
   board_print(bd);

   while ( 1 ) {

      if ( get_turn(bd) == WHITE ) printf("W: ");
      else printf("B: ");
      fflush(stdout);
      fgets(buffer,BUFLEN,stdin);
      trim(buffer);

      if ( parse(buffer, "quit","q") ) break;
      if ( parse(buffer,"print","p") ) board_print(bd);
      if ( parse(buffer,"flags","f") ) board_print_flags(bd);
      if ( parse(buffer,"reset","r") ) {
         board_init(bd);
         board_print(bd);
      }
      if ( parse(buffer,"clear","c") ) {
         board_clear(bd);
         board_print(bd);
      }

      // Evaluate moves at position
      if ( strlen(buffer) == 2 ) {
         sz = 0;
         pos = ind(buffer);
         get_moves(bd,pos,moves,&sz);
         board_mark_print(bd,moves,sz);
         print_moves(moves,sz);
      }

      // -x0  Removes a piece
      if ( strlen(buffer) == 3 && buffer[0] == '-' ) {
         pos = ind(buffer + 1);
         if ( pos >= 0 ) {
            if ( remove_piece(bd,pos) ) board_print(bd);
         }
      }

      // Insert piece
      if ( strlen(buffer) == 4 && buffer[0] == '+' ) {
         if ( is_piece(buffer[1]) ) {
            pos = ind(buffer + 2);
            if ( pos >= 0 ) {
               place_piece(bd,pos,buffer[1]);
               board_print(bd);
            }
         }
      }

      // Move piece (valid move)
      if ( strlen(buffer) == 4 && buffer[0] >= 'a' && buffer[0] <= 'h' ) {
         a = ind(buffer);
         b = ind(buffer + 2);
         if ( a >= 0 && b >= 0 ) {
            get_moves(bd,pos,moves,&sz);
            e = make_move(bd,a,b,moves,sz);
            if ( !e ) board_print(bd);
            else print_move_error(e);
         }
      }

      // Move piece (force move)
      if ( strlen(buffer) == 5 && buffer[0] == '!' ) {
         a = ind(buffer + 1);
         b = ind(buffer + 3);
         if ( a >= 0 && b >= 0 ) {
            force_move(bd,a,b);
            board_print(bd);
         }
      }

   }

   free(bd);
   return 0;
}

