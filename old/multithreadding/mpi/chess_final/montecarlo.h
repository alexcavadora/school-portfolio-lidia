#ifndef MONTE_CARLO_CHESS_H
#define MONTE_CARLO_CHESS_H

#include "chess.h"

// Configuration for Monte Carlo simulations
#define MC_SIMULATIONS 100000  // Number of random game simulations
#define MC_MAX_MOVES 10000     // Maximum moves in a simulation

// Structure to store move evaluation
typedef struct {
    int start_pos;       // Starting position of the move
    int end_pos;         // Ending position of the move
    float win_rate;      // Calculated win rate for this move
    int total_simulations;  // Total number of simulations for this move
} mc_move_eval_t;

// Evaluate moves using Monte Carlo method
int monte_carlo_move(board_t* bd, int* best_start, int* best_end);

// Simulate a random game from current board state
int simulate_random_game(board_t* initial_board);

// Evaluate a board state (simple heuristic)
float evaluate_board(const board_t* bd);

#endif /* MONTE_CARLO_CHESS_H */