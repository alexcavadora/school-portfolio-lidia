#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "chess.h"
#include "montecarlo.h"

// Initialize random seed
void init_random_seed() {
    static int seed_initialized = 0;
    if (!seed_initialized) {
        srand(time(NULL));
        seed_initialized = 1;
    }
}

// Simple board evaluation function
float evaluate_board(const board_t* bd) {
    float score = 0;
    color_t turn_color = get_turn(bd);
    color_t opponent_color = (turn_color == WHITE) ? BLACK : WHITE;

    for (int pos = 0; pos < 64; pos++) {
        piece_t piece = get_piece(bd, pos);
        color_t piece_color = get_color(bd, pos);

        if (piece_color == turn_color) {
            // Positive score for our pieces
            switch (piece) {
                case PAWN:   score += 1.0; break;
                case KNIGHT: score += 3.0; break;
                case BISHOP: score += 3.0; break;
                case ROOK:   score += 5.0; break;
                case QUEEN:  score += 9.0; break;
                case KING:   score += 0.5; break;
                default: break;
            }
        } else if (piece_color == opponent_color) {
            // Negative score for opponent's pieces
            switch (piece) {
                case PAWN:   score -= 1.0; break;
                case KNIGHT: score -= 3.0; break;
                case BISHOP: score -= 3.0; break;
                case ROOK:   score -= 5.0; break;
                case QUEEN:  score -= 9.0; break;
                case KING:   score -= 0.5; break;
                default: break;
            }
        }
    }

    // Bonus for not being in check
    if (!test_check(bd)) {
        score += 1.0;
    }

    return score;
}

// Simulate a random game from current board state
int simulate_random_game(board_t* initial_board) {
    board_t board;
    int moves[MOVLEN];
    int sz;

    // Create a copy of the initial board
    board_copy(initial_board, &board);

    // Simulate up to MC_MAX_MOVES random moves
    for (int i = 0; i < MC_MAX_MOVES; i++) {
        // Check for game-ending conditions
        if (test_check(&board)) {
            // If current turn's king is in check, this is likely a loss
            return (get_turn(&board) == WHITE) ? -1 : 1;
        }

        // Get all possible moves for current turn
        sz = 0;
        int valid_moves_found = 0;

        int pos = -1;
        for (pos = 0; pos < 64; pos++) {
            if (get_color(&board, pos) == get_turn(&board)) {
                sz = 0;
                if (get_moves(&board, pos, moves, &sz) && sz > 0) {
                    valid_moves_found = 1;
                    break;
                }
            }
        }

        // If no valid moves, game is over
        if (!valid_moves_found) {
            return (get_turn(&board) == WHITE) ? -1 : 1;
        }

        // Select a random move
        int move_index = rand() % sz;
        int start_pos = pos;
        int end_pos = moves[move_index];

        // Apply the move
        int move_result = make_move(&board, start_pos, end_pos, moves, sz);
        if (move_result != 0) {
            // If move is invalid, game is over
            return (get_turn(&board) == WHITE) ? -1 : 1;
        }
    }

    // If max moves reached, consider it a draw
    return 0;
}

// Evaluate moves using Monte Carlo method
int monte_carlo_move(board_t* bd, int* best_start, int* best_end) {
    init_random_seed();

    // Collect all possible moves for current turn
    int all_moves[MOVLEN];
    int all_moves_count = 0;
    mc_move_eval_t move_evaluations[MOVLEN];
    int unique_move_count = 0;

    // Collect all valid moves
    for (int start_pos = 0; start_pos < 64; start_pos++) {
        if (get_color(bd, start_pos) == get_turn(bd)) {
            int moves[MOVLEN];
            int sz = 0;
            if (get_moves(bd, start_pos, moves, &sz)) {
                for (int j = 0; j < sz; j++) {
                    // Create a copy of the board to test the move
                    board_t test_board;
                    board_copy(bd, &test_board);

                    // Try the move
                    if (make_move(&test_board, start_pos, moves[j], moves, sz) == 0) {
                        // Store this move
                        all_moves[all_moves_count++] = start_pos;
                        all_moves[all_moves_count++] = moves[j];

                        // Initialize move evaluation
                        move_evaluations[unique_move_count].start_pos = start_pos;
                        move_evaluations[unique_move_count].end_pos = moves[j];
                        move_evaluations[unique_move_count].win_rate = 0;
                        move_evaluations[unique_move_count].total_simulations = 0;
                        unique_move_count++;
                    }
                }
            }
        }
    }

    // If no moves available, return failure
    if (unique_move_count == 0) {
        return 0;
    }

    // Run Monte Carlo simulations
    for (int sim = 0; sim < MC_SIMULATIONS; sim++) {
        // Choose a random move to simulate
        int move_index = rand() % unique_move_count;

        // Create a copy of the board to test the move
        board_t sim_board;
        board_copy(bd, &sim_board);

        // Apply the move
        if (make_move(&sim_board, 
            move_evaluations[move_index].start_pos, 
            move_evaluations[move_index].end_pos, 
            all_moves, all_moves_count) == 0) {
            
            // Simulate the game from this point
            int game_result = simulate_random_game(&sim_board);

            // Update move statistics
            move_evaluations[move_index].total_simulations++;
            
            // Adjust win rate based on game result
            if (game_result > 0) {
                move_evaluations[move_index].win_rate += 1.0;
            } else if (game_result < 0) {
                move_evaluations[move_index].win_rate -= 1.0;
            }
        }
    }

    // Find the best move
    float best_win_rate = -1000000.0;
    int best_move_index = 0;

    for (int i = 0; i < unique_move_count; i++) {
        if (move_evaluations[i].total_simulations > 0) {
            float normalized_win_rate = 
                move_evaluations[i].win_rate / move_evaluations[i].total_simulations;

            if (normalized_win_rate > best_win_rate) {
                best_win_rate = normalized_win_rate;
                best_move_index = i;
            }
        }
    }

    // Return the best move
    *best_start = move_evaluations[best_move_index].start_pos;
    *best_end = move_evaluations[best_move_index].end_pos;

    return 1;
}