#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

#define WINDOW_WIDTH 30
#define WINDOW_HEIGHT 15
#define PLAYER_SYMBOL '@'
#define ENEMY_SYMBOL 'X'
#define PILLAR_SYMBOL '#'
#define HIDDEN_PLAYER_SYMBOL ' '
#define ENEMY_VISION_RANGE 3

// Function to configure terminal for immediate input
void configureTerminal() {
    struct termios tty;
    tcgetattr(STDIN_FILENO, &tty);
    tty.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);
    fcntl(STDIN_FILENO, F_SETFL, fcntl(STDIN_FILENO, F_GETFL) | O_NONBLOCK);
}

// Function to restore terminal settings
void restoreTerminal() {
    struct termios tty;
    tcgetattr(STDIN_FILENO, &tty);
    tty.c_lflag |= ICANON | ECHO;
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);
    fcntl(STDIN_FILENO, F_SETFL, fcntl(STDIN_FILENO, F_GETFL) & ~O_NONBLOCK);
}

int main() {
    int playerX, playerY, enemyX, enemyY;
    int pillars[4][2] = {
        {5, 5}, {10, 10}, {20, 5}, {25, 10}
    };
    int gameTime = 0;
    int gameWon = 0;
    char gameWindow[WINDOW_HEIGHT][WINDOW_WIDTH];
    char input;

    // Configure terminal for game input
    configureTerminal();

    // Seed the random number generator
    srand(time(NULL));

    // Initialize player and enemy positions
    playerX = rand() % WINDOW_WIDTH;
    playerY = rand() % WINDOW_HEIGHT;
    enemyX = rand() % WINDOW_WIDTH;
    enemyY = rand() % WINDOW_HEIGHT;

    // Initialize the game window with pillars
    for (int y = 0; y < WINDOW_HEIGHT; y++) {
        for (int x = 0; x < WINDOW_WIDTH; x++) {
            gameWindow[y][x] = HIDDEN_PLAYER_SYMBOL;
        }
    }

    for (int i = 0; i < 4; i++) {
        gameWindow[pillars[i][1]][pillars[i][0]] = PILLAR_SYMBOL;
    }

    printf("\033[?25l"); // Hide cursor

    while (1) {
        // Clear the screen and move cursor to top-left
        printf("\033[H");

        // Print the game window
        for (int y = 0; y < WINDOW_HEIGHT; y++) {
            for (int x = 0; x < WINDOW_WIDTH; x++) {
                if (x == playerX && y == playerY) {
                    printf("%c", PLAYER_SYMBOL);
                } else if (x == enemyX && y == enemyY) {
                    printf("%c", ENEMY_SYMBOL);
                } else if (abs(x - enemyX) <= ENEMY_VISION_RANGE &&
                         abs(y - enemyY) <= ENEMY_VISION_RANGE) {
                    printf("\033[36m~\033[0m"); // Cyan vision range
                } else {
                    printf("%c", gameWindow[y][x]);
                }
            }
            printf("\n");
        }

        // Print game info
        printf("\nTime: %d/20 seconds", gameTime);
        printf("\nControls: WASD to move, Q to quit");

        // Check if the player is found
        if (abs(playerX - enemyX) <= 1 && abs(playerY - enemyY) <= 1) {
            printf("\nYou were found! Game over.\n");
            break;
        }

        // Get player input
        if (read(STDIN_FILENO, &input, 1) > 0) {
            switch (input) {
                case 'w':
                    if (playerY > 0 && gameWindow[playerY-1][playerX] != PILLAR_SYMBOL) {
                        playerY--;
                    }
                    break;
                case 's':
                    if (playerY < WINDOW_HEIGHT - 1 && gameWindow[playerY+1][playerX] != PILLAR_SYMBOL) {
                        playerY++;
                    }
                    break;
                case 'a':
                    if (playerX > 0 && gameWindow[playerY][playerX-1] != PILLAR_SYMBOL) {
                        playerX--;
                    }
                    break;
                case 'd':
                    if (playerX < WINDOW_WIDTH - 1 && gameWindow[playerY][playerX+1] != PILLAR_SYMBOL) {
                        playerX++;
                    }
                    break;
                case 'q':
                    gameWon = -1;
                    goto cleanup;
            }
        }

        // Update the enemy's position if player is in vision range
        int playerVisible = 1;
        // Check if any pillar blocks line of sight
        for (int i = 0; i < 4; i++) {
            if (abs(pillars[i][0] - enemyX) <= 1 &&
                abs(pillars[i][1] - enemyY) <= 1) {
                playerVisible = 0;
                break;
            }
        }

        if (playerVisible &&
            abs(playerX - enemyX) <= ENEMY_VISION_RANGE &&
            abs(playerY - enemyY) <= ENEMY_VISION_RANGE) {
            if (playerX < enemyX) enemyX--;
            else if (playerX > enemyX) enemyX++;
            if (playerY < enemyY) enemyY--;
            else if (playerY > enemyY) enemyY++;
        }

        // Increase the game time
        gameTime++;

        // Check if the player has won
        if (gameTime >= 2000) {
            printf("\nYou win! You survived for 20 seconds!\n");
            gameWon = 1;
            break;
        }

        // Wait for a short time
        usleep(100000);  // 0.1 second delay
    }

cleanup:
    // Show cursor again
    printf("\033[?25h");

    // Restore terminal settings
    restoreTerminal();

    // Print final message
    if (gameWon == 1) {
        printf("\nCongratulations! You won!\n");
    } else if (gameWon == 0) {
        printf("\nGame Over! The enemy caught you!\n");
    } else {
        printf("\nGame quit by player.\n");
    }

    return 0;
}
