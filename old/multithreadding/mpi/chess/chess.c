// chess.c

#include "chess.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdbool.h>

#define RESET "\033[0m"
#define WHITE_SQUARE "\033[48;5;223m"
#define BLACK_SQUARE "\033[48;5;130m"
#define WHITE_PIECE "\033[97m"
#define BLACK_PIECE "\033[30m"

int move_get(char x, char y, int* row, int* col) {
    if (x < 'a' || x > 'h' || y < '1' || y > '8') return 0;  // Check valid range

    *row = 7 - (y - '1');  // Calculate row index (0-7 from top to bottom)
    *col = x - 'a';             // Calculate column index (0-7 from 'a' to 'h')

    return 1;  // Success
}
// Helper functions for piece and square analysis
bool is_empty(board* bd, int row, int col) {
    return bd->square[row][col] == '.';
}

bool is_white_piece(char piece) {
    return isupper(piece);
}

bool is_black_piece(char piece) {
    return islower(piece) && piece != '.';
}

bool is_same_color(char piece1, char piece2) {
    return (is_white_piece(piece1) && is_white_piece(piece2)) ||
           (is_black_piece(piece1) && is_black_piece(piece2));
}

bool is_valid_position(int row, int col) {
    return row >= 0 && row < 8 && col >= 0 && col < 8;
}

bool can_capture(char attacker, char target) {
    return target != '.' && !is_same_color(attacker, target);
}


bool is_diagonal_move(int start_row, int start_col, int end_row, int end_col) {
    return abs(end_row - start_row) == abs(end_col - start_col);
}

bool is_straight_move(int start_row, int start_col, int end_row, int end_col) {
    return start_row == end_row || start_col == end_col;
}

bool is_knight_move(int start_row, int start_col, int end_row, int end_col) {
    int row_diff = abs(end_row - start_row);
    int col_diff = abs(end_col - start_col);
    return (row_diff == 2 && col_diff == 1) || (row_diff == 1 && col_diff == 2);
}

// Path checking
bool is_path_clear(board* bd, int start_row, int start_col, int end_row, int end_col) {
    int row_step = (end_row > start_row) ? 1 : (end_row < start_row) ? -1 : 0;
    int col_step = (end_col > start_col) ? 1 : (end_col < start_col) ? -1 : 0;

    int row = start_row + row_step;
    int col = start_col + col_step;

    while (row != end_row || col != end_col) {
        if (!is_empty(bd, row, col)) {
            return false;
        }
        row += row_step;
        col += col_step;
    }

    return true;
}

// Piece-specific movement validation
bool bishop_valid(board* bd, int start_row, int start_col, int end_row, int end_col) {
    return is_diagonal_move(start_row, start_col, end_row, end_col) &&
           is_path_clear(bd, start_row, start_col, end_row, end_col);
}

bool rook_valid(board* bd, int start_row, int start_col, int end_row, int end_col) {
    return is_straight_move(start_row, start_col, end_row, end_col) &&
           is_path_clear(bd, start_row, start_col, end_row, end_col);
}

bool queen_valid(board* bd, int start_row, int start_col, int end_row, int end_col) {
    return (is_straight_move(start_row, start_col, end_row, end_col) ||
            is_diagonal_move(start_row, start_col, end_row, end_col)) &&
           is_path_clear(bd, start_row, start_col, end_row, end_col);
}

bool knight_valid(board* bd, int start_row, int start_col, int end_row, int end_col) {
    return is_knight_move(start_row, start_col, end_row, end_col);
}

bool king_valid(board* bd, int start_row, int start_col, int end_row, int end_col) {
    return abs(end_row - start_row) <= 1 && abs(end_col - start_col) <= 1;
}

bool pawn_valid(board* bd, char x1, char y1, char x2, char y2, char color) {
    int start_row, start_col, end_row, end_col;

    if (!move_get(x1, y1, &start_row, &start_col) ||
        !move_get(x2, y2, &end_row, &end_col)) {
        return false;
    }

    int direction = (color == 'w') ? -1 : 1;
    int initial_row = (color == 'w') ? 6 : 1;

    if (start_col == end_col && is_empty(bd, end_row, end_col)) {
        // Single square advance
        if (end_row == start_row + direction) {
            return true;
        }
        // Initial two-square advance
        if (start_row == initial_row &&
            end_row == start_row + 2 * direction &&
            is_empty(bd, start_row + direction, start_col)) {
            bd->last_pawn_double_move = (color == 'w') ? x1 : 0; // Track for en passant
            return true;
        }
    }

    // Diagonal capture
    if (abs(start_col - end_col) == 1 && end_row == start_row + direction) {
        // Regular capture
        if (!is_empty(bd, end_row, end_col) &&
            ((color == 'w' && is_black_piece(bd->square[end_row][end_col])) ||
             (color == 'b' && is_white_piece(bd->square[end_row][end_col])))) {
            return true;
        }

        // En passant
        char last_move_col = bd->last_pawn_double_move - 'a';
        if (color == 'w' && start_row == 3 && // White capturing black pawn
            bd->last_pawn_double_move && end_col == last_move_col &&
            bd->square[start_row][end_col] == 'p') {
            bd->square[start_row][end_col] = '.'; // Remove captured pawn
            return true;
        }
        if (color == 'b' && start_row == 4 && // Black capturing white pawn
            bd->last_pawn_double_move && end_col == last_move_col &&
            bd->square[start_row][end_col] == 'P') {
            bd->square[start_row][end_col] = '.'; // Remove captured pawn
            return true;
        }
    }

    return false;
}

void board_init(board* bd) {
    // Initialize pieces
    const char back_rank[] = {'r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'};

    for (int i = 0; i < 8; i++) {
        bd->square[0][i] = back_rank[i];        // Black back rank
        bd->square[1][i] = 'p';                 // Black pawns
        bd->square[6][i] = 'P';                 // White pawns
        bd->square[7][i] = toupper(back_rank[i]); // White back rank

        // Empty squares
        for (int j = 2; j < 6; j++) {
            bd->square[j][i] = '.';
        }
    }

    // Initialize game state
    bd->last_pawn_double_move = 0;
}

int move_piece(board* bd, char x1, char y1, char x2, char y2) {
    int start_row, start_col, end_row, end_col;

    if (!move_get(x1, y1, &start_row, &start_col) ||
        !move_get(x2, y2, &end_row, &end_col)) {
        return 0;
    }

    char piece = bd->square[start_row][start_col];
    if (is_empty(bd, start_row, start_col)) {
        return 0;
    }

    // Check if trying to capture own piece
    if (!is_empty(bd, end_row, end_col) &&
        is_same_color(piece, bd->square[end_row][end_col])) {
        return 0;
    }

    // Validate piece-specific movement
    bool valid_move = false;
    char piece_type = tolower(piece);
    char color = is_white_piece(piece) ? 'w' : 'b';

    switch (piece_type) {
        case 'p':
            valid_move = pawn_valid(bd, x1, y1, x2, y2, color);
            break;
        case 'r':
            valid_move = rook_valid(bd, start_row, start_col, end_row, end_col);
            break;
        case 'n':
            valid_move = knight_valid(bd, start_row, start_col, end_row, end_col);
            break;
        case 'b':
            valid_move = bishop_valid(bd, start_row, start_col, end_row, end_col);
            break;
        case 'q':
            valid_move = queen_valid(bd, start_row, start_col, end_row, end_col);
            break;
        case 'k':
            valid_move = king_valid(bd, start_row, start_col, end_row, end_col);
            break;
    }

    if (!valid_move) {
        return 0;
    }

    // Clear en passant tracking if it's not a pawn double move
    if (piece_type != 'p' || abs(end_row - start_row) != 2) {
        bd->last_pawn_double_move = 0;
    }

    // Make the move
    bd->square[end_row][end_col] = piece;
    bd->square[start_row][start_col] = '.';

    return 1;
}

const char* get_piece_symbol(char piece) {
    switch (piece) {
        case 'p': return BLACK_PIECE "♟";
        case 'r': return BLACK_PIECE "♜";
        case 'n': return BLACK_PIECE "♞";
        case 'b': return BLACK_PIECE "♝";
        case 'q': return BLACK_PIECE "♛";
        case 'k': return BLACK_PIECE "♚";
        case 'P': return WHITE_PIECE "♟";
        case 'R': return WHITE_PIECE "♜";
        case 'N': return WHITE_PIECE "♞";
        case 'B': return WHITE_PIECE "♝";
        case 'Q': return WHITE_PIECE "♛";
        case 'K': return WHITE_PIECE "♚";
        case '.': return " ";
        default: return " ";
    }
}

void board_print(board* bd) {
    printf("\n    a  b  c  d  e  f  g  h\n");
    for (int i = 0; i < 8; i++) {
        printf(" %d ", 8 - i);
        for (int j = 0; j < 8; j++) {
            const char* square_color = (i + j) % 2 == 0 ? WHITE_SQUARE : BLACK_SQUARE;
            printf("%s %s %s", square_color, get_piece_symbol(bd->square[i][j]), RESET);
        }
        printf(" %d\n", 8 - i);
    }
    printf("    a  b  c  d  e  f  g  h\n\n");
}
