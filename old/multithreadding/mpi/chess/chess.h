#ifndef CHESS_H
#define CHESS_H
#include <stdbool.h>
#define BOARD_SIZE 8

typedef struct {
    char square[BOARD_SIZE][BOARD_SIZE];
    char last_pawn_double_move;
} board;

void board_init(board* bd);
void board_print(board* bd);
int move_get(char x, char y, int* row, int* col);
int move_piece(board* bd, char x1, char y1, char x2, char y2);

bool pawn_valid(board* bd, char x1, char y1, char x2, char y2, char color);
bool rook_valid(board* bd, int start_row, int start_col, int end_row, int end_col);
bool knight_valid(board* bd, int start_row, int start_col, int end_row, int end_col);
bool bishop_valid(board* bd, int start_row, int start_col, int end_row, int end_col);
bool queen_valid(board* bd, int start_row, int start_col, int end_row, int end_col);
bool king_valid(board* bd, int start_row, int start_col, int end_row, int end_col);

#endif // CHESS_H
