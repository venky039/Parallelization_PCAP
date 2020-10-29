#include "board.c"
#include <mpi.h>

#define min(a,b) (a < b ? a : b)


int rank, size;
MPI_Status status;

struct job_struct {
	int alpha;
	symbol_t symbol;
	board_t board;
};

// EURISTHIC: win as fast as we can
int get_score(board_t* board, int depth, symbol_t symbol) {
	symbol_t result = winner(board);
	
	if(result == symbol) {
		return N * M + 10 - depth;
	} else if(result != EMPTY && result != NO_WINNER) {
		return -(N * M) - 10 + depth;
	} else if(result == NO_WINNER) {
		return 1;
	}

	return 0;
}

int move(board_t* board, symbol_t symbol, int depth, int alpha, int beta) {
	int n, i;
	move_t* max_move;
	int score = get_score(board, depth, symbol);

	if(score != 0) {
		return score;
	}

	move_t** moves = get_all_possible_moves(board, symbol, &n);
	for(i = 0; i < n; i++) {
		put_symbol(board, symbol, moves[i]);
		score = -move(board, other_symbol(symbol), depth + 1, -beta, -alpha);
		clear_symbol(board, moves[i]);

		if(score > alpha) {
			alpha = score;
			max_move = moves[i];
		}

		if(alpha >= beta) {
			break;
		}
	}

	for(i = 0; i < n; i++) {
		free(moves[i]);
	}

	free(moves);
	
	return alpha;
}

int main(int argc, char* argv[]) {

	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &size);

	symbol_t result;
	symbol_t current_symbol = X;
	board_t* board = create_board();
	int score;
	symbol_t done_symbol = 2;

	struct job_struct* job = malloc(sizeof(struct job_struct));
	struct job_struct jobs[200];

	if(rank == 0) 
	{
		int i, n, best_score_index, best_score;
		move_t** moves;
		int finished[size];
		int current_move[size];

		for(i = 0; i < size; i++) {
			finished[i] = 0;
		}

		

		while(1) 
		{
			best_score = -9999;
			for(int i=0;i<200;i++)
			jobs[i].alpha = best_score;

			if(current_symbol==0)
			printf(COLOR_RESET"\t\tCPU to move \n");
		else
			printf(COLOR_RESET"\t\tPlayer to move \n");


			moves = get_all_possible_moves(board, current_symbol, &n);
			if(n==1){printf(COLOR_RED"Draw No more Moves Left\n"COLOR_RESET);MPI_Finalize();exit(0);}

			if((int) current_symbol==0)
			{

				

			// pass one task to each available process
			for(i = 0; i < min(n, size-1); i++) 
			{
				//printf("send move %i to %i\n", i, i + 1);

				put_symbol(board, current_symbol, moves[i]);

				jobs[i].board = *board;
				jobs[i].symbol = other_symbol(current_symbol);
				
				MPI_Send(&jobs[i], sizeof(struct job_struct), MPI_BYTE, i + 1, 1, MPI_COMM_WORLD);
				clear_symbol(board, moves[i]);

				current_move[i+1] = i;
			}

			// if there are more moves to make than processes
			for(i = size - 1; i < n; i++) 
				{
					//if(i==size-1)printf("thode jyada bana lete processes");
				MPI_Recv(&score, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &status);
				//printf("received score %i from %i\n", score, status.MPI_SOURCE);

				if(score > best_score) {
					best_score = score;
					best_score_index = current_move[status.MPI_SOURCE];
					jobs[i].alpha = best_score;
				}

				put_symbol(board, current_symbol, moves[i]);
				
				jobs[i].board = *board;
				jobs[i].symbol = other_symbol(current_symbol);

				MPI_Send(&jobs[i], sizeof(struct job_struct), MPI_BYTE, status.MPI_SOURCE, 1, MPI_COMM_WORLD);

				clear_symbol(board, moves[i]);
				//printf("send move %i to %i\n", i, status.MPI_SOURCE);


				current_move[status.MPI_SOURCE] = i;
			}

			// if there are more processes than moves then the rest of processes must be finished
			
			for(i = n + 1; i < size; i++) {
				jobs[i].symbol = done_symbol;
				if(finished[i]) break;

			//	printf("finishing %i\n", i);
				MPI_Send(&jobs[i], sizeof(struct job_struct), MPI_BYTE, i, 1, MPI_COMM_WORLD);
				finished[i] = 1;
			}

			// wait for the rest of results
			for(i = 0; i < min(n, size - 1); i++) {
				MPI_Recv(&score, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &status);
				if(score > best_score) {
					best_score = score;
					best_score_index = current_move[status.MPI_SOURCE];
				}
			//	printf("received score %i from %i\n", score, status.MPI_SOURCE);
			}

			put_symbol(board, current_symbol, moves[best_score_index]);

			print_board(board);

			for(i = 0; i < n; i++) {
				free(moves[i]);
			}

			free(moves);

			result = winner(board);
			if(result != EMPTY) {
				break;
			}
			
			}
			else
			{
				int playMove;
				print_board_player(board);
				printf("\nenter move accordingly \n");
				scanf("%d",&playMove);
					put_symbol(board, current_symbol, moves[playMove-1]);

					print_board(board);

			}
			current_symbol = 1 - current_symbol;

		}

		
		for(i = 1; i < size; i++) {
			jobs[i].symbol = done_symbol;
			if(finished[i]) break;

		//	printf("finishing %i\n", i);
			MPI_Send(&jobs[i], sizeof(struct job_struct), MPI_BYTE, i, 1, MPI_COMM_WORLD);
		}

		if(result==0)
		printf(COLOR_GREEN"\t\tCPU Wins\n"COLOR_RESET);
	else if(result==1)
		printf(COLOR_GREEN"\t\tPlayer Wins\n"COLOR_RESET);
	} else {
		while(1) {
			
			MPI_Recv(job, sizeof(struct job_struct), MPI_BYTE, 0, 1, MPI_COMM_WORLD, &status);
			if(job->symbol == done_symbol) {
				break;
			}
			score = -move(&job->board, job->symbol, 0, -9999, -job->alpha);
			MPI_Send(&score, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
		}
	
	}

	MPI_Finalize();
	return 0;
}