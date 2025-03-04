#include "chess.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

void clear_screen() {
    printf("\033[H\033[J");
}

void clear_input_buffer() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

int main()
{
    int moves = 0;
    board bd;
    board_init(&bd);
    char x1, x2, y1, y2;
    while (1) {
        clear_screen();
        board_print(&bd);
        if (moves %2 == 0) printf("White to move:\n");
        else printf("nigga to move...\n");
        printf("Turn %i.\nEnter your move (e.g., e2 e4) or 'exit' to quit: ", moves);
        char input[10];
        if (fgets(input, sizeof(input), stdin) == NULL) {
            break;
        }

        // Check for exit
        if (strncmp(input, "exit", 4) == 0) {
            break;
        }

        // Parse move input
        if (strlen(input) >= 5) {
            x1 = input[0];
            y1 = input[1];
            x2 = input[3];
            y2 = input[4];

            if (!move_piece(&bd, x1, y1, x2, y2)) {
                printf("Invalid move from %c%c to %c%c.\n", x1, y1, x2, y2);
                printf("Press Enter to continue...");
                clear_input_buffer();
            }
            else{
                moves++;
            }
        } else {
            printf("Invalid input format. Please use format 'e2 e4'.\n");
            printf("Press Enter to continue...");
            clear_input_buffer();
        }


    }
}
