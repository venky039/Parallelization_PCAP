//#include "board.h"
//#ifndef __BOARD_DEF


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define X 1
#define EMPTY 10
#define NO_WINNER 20
#define COLOR_RED "\x1b[31m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_YELLOW "\x1b[33m"
#define COLOR_BLUE "\x1b[34m"
#define COLOR_CYAN "\x1b[36m"
#define COLOR_RESET "\x1b[0m"

#define N 3
#define M N

typedef unsigned char symbol_t;

typedef struct board {
	symbol_t m[N][M];
	unsigned short n_empty;
} board_t;

typedef struct move {
	unsigned short i, j;
} move_t;

board_t* create_board();
void put_symbol(board_t*, symbol_t, move_t*);
void clear_symbol(board_t*, move_t*);
symbol_t winner(board_t*);
void print_board(board_t*);
void print_board_player(board_t*);
move_t** get_all_possible_moves(board_t*, symbol_t, int*);
symbol_t other_symbol(symbol_t);

//#endif

board_t* create_board() {
	int i, j;
	board_t* board = (board_t*) malloc(sizeof(board_t));
	for(i = 0; i < N; i++) {
		for(j = 0; j < M; j++) {
			board->m[i][j] = EMPTY;
		}
	}
	board->n_empty = N * M;

	return board;
}

void put_symbol(board_t* board, symbol_t symbol, move_t* move) {
	board->m[move->i][move->j] = symbol;
	board->n_empty --;
}

void clear_symbol(board_t* board, move_t* move) {
	board->m[move->i][move->j] = EMPTY;
	board->n_empty ++;
}

symbol_t winner(board_t* b) {
	int i, j;
	symbol_t sym;
	int equal;

	// check on lines
	for(i = 0; i < N; i++) {
		equal = 1;
		sym = b->m[i][0];
		if(sym != EMPTY) {
			for(j = 1; j < M; j++) {
				if(b->m[i][j] != sym) {
					equal = 0;
					break;
				}
			}

			if(equal == 1) {
				return sym;
			}
		}
	
	}

	// check on columns
	for(i = 0; i < M; i++) {
		equal = 1;
		sym = b->m[0][i];
		if(sym != EMPTY) {
			for(j = 1; j < N; j++) {
				if(b->m[j][i] != sym) {
					equal = 0;
					break;
				}
			}

			if(equal == 1) {
				return sym;
			}
		}
	
	}
	
	// main diagonal	
	equal = 1;
	sym = b->m[0][0];
	if(sym != EMPTY) {
		for(i = 1; i < N; i++) {
			if(b->m[i][i] != sym) {
				equal = 0;
				break;
			}
		}

		if(equal == 1) {
			return sym;
		}
	}

	// secondary diagonal
	equal = 1;
	sym = b->m[0][M-1];
	if(sym != EMPTY) {
		for(i = 1; i < N; i++) {
			if(b->m[i][M-i-1] != sym) {
				equal = 0;
				break;
			}
		}

		if(equal == 1) {
			return sym;
		}
	}

	if(b->n_empty == 0) {
		return NO_WINNER;
	}

	return EMPTY;
}

void print_board(board_t* board) {
	int i, j;
	for(i = 0; i < N; i++) {
		printf("\t\t");
		for(j = 0; j < M; j++) {
			if(board->m[i][j] == X) {
				printf(COLOR_YELLOW"   X   "COLOR_RESET);
			} else if(board->m[i][j] == 0) {
				printf(COLOR_YELLOW"   O   "COLOR_RESET);
			} else {
				printf(COLOR_YELLOW"   -   "COLOR_RESET);
			}
			if(j<M-1)
			printf(COLOR_YELLOW"|");
		}
		printf(COLOR_YELLOW"\n\t\t  -------------------\n");
	}
}

void print_board_player(board_t* board) {
	int i, j;
	int qw=1;
	for(i = 0; i < N; i++) {
		printf("\t\t");
		for(j = 0; j < M; j++) {
			
			if(board->m[i][j] == X) {
				printf(COLOR_YELLOW"   X   "COLOR_RESET);
			} else if(board->m[i][j] == 0) {
				printf(COLOR_YELLOW"   Y   "COLOR_RESET);
			} else {
				printf(COLOR_YELLOW"   %d   "COLOR_RESET,qw++);
			}
			if(j<M-1)
			printf(COLOR_YELLOW"|"COLOR_RESET);
		}
		printf(COLOR_YELLOW"\n\t\t  -------------------\n"COLOR_RESET);
	}
}

move_t** get_all_possible_moves(board_t* board, symbol_t symbol, int* n) {
	int i,j;

	move_t** list = (move_t**) malloc(board->n_empty * sizeof(move_t*));
	*n = 0;

	for(i = 0; i < N; i++) {
		for(j = 0; j < M; j++) {
			if(board->m[i][j] == EMPTY) {
				list[(*n)] = (move_t*) malloc(sizeof(move_t));
				list[(*n)]->i = i;
				list[(*n)]->j = j;
				(*n) ++;
			}
		}

	}
	return list;
}

symbol_t other_symbol(symbol_t symbol) {
	return 1 - symbol;
}