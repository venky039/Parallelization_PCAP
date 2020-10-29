#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
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

typedef struct job_struct {
    int alpha;
    symbol_t symbol;
    board_t board;
}Sjob;








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

__device__ __host__ void put_symbol(board_t* board, symbol_t symbol, move_t* move) {
    board->m[move->i][move->j] = symbol;
    board->n_empty --;
}

__device__ __host__ void clear_symbol(board_t* board, move_t* move) {
    board->m[move->i][move->j] = EMPTY;
    board->n_empty ++;
}

__device__ __host__  symbol_t winner(board_t* b) {
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
                printf(COLOR_YELLOW"   X   ");
            } else if(board->m[i][j] == 0) {
                printf(COLOR_YELLOW"   O   ");
            } else {
                printf(COLOR_YELLOW"   -   ");
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
                printf(COLOR_YELLOW"   X   ");
            } else if(board->m[i][j] == 0) {
                printf(COLOR_YELLOW"   Y   ");
            } else {
                printf(COLOR_YELLOW"   %d   ",qw++);
            }
            if(j<M-1)
            printf(COLOR_YELLOW"|");
        }
        printf(COLOR_YELLOW"\n\t\t  -------------------\n");
    }
}

 __device__ __host__  move_t** get_all_possible_moves(board_t* board, symbol_t symbol, int* n) {
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

__device__ __host__  symbol_t other_symbol(symbol_t symbol) {
    return 1 - symbol;
}



  

__device__ __host__ int get_score(board_t* board, int depth, symbol_t symbol) {
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

__device__ __host__ int move(board_t* board, symbol_t symbol, int depth, int alpha, int beta) {
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
__global__ void GetScoreKernel(Sjob *a,int* sc) {
    
    int ci = threadIdx.x;
    sc[ci] = -move(&(a[ci].board), a[ci].symbol, 0, -9999, -(a[ci].alpha));

}


        int main()
        {

              Sjob* d_jobs;
            int * d_scores;
            symbol_t result;
    symbol_t current_symbol = X;
    board_t* board = create_board();
    int score;
  //  symbol_t done_symbol = 2;
            int  n, best_score_index, best_score;
        move_t** moves;

        int current_move[100];

       

struct job_struct* job = (Sjob *)malloc(sizeof(struct job_struct));
        Sjob jobs[200];

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
            

            if((int) current_symbol==0)
            {
            if(n==0){printf(COLOR_RED"\t\tDraw\n No more Moves Left\n");exit(0);}
               

            // pass one task to each available process
            for(int i = 0; i < n; i++)
            {
              //  printf("send move %i to %i\n", i, i + 1);

                put_symbol(board, current_symbol, moves[i]);

                jobs[i].board = *board;
                jobs[i].symbol = other_symbol(current_symbol);
               
                
                clear_symbol(board, moves[i]);

                current_move[i+1] = i;
            }

            // if there are more moves to make than processes
          

            cudaMalloc((void **)&d_jobs,100*sizeof(Sjob));
            cudaMalloc((void **)&d_scores,100*sizeof(int));
            cudaMemcpy(d_jobs,jobs,n*sizeof(Sjob),cudaMemcpyHostToDevice);

            GetScoreKernel<<<1,n>>>(d_jobs,d_scores);
            int scores[n];
            cudaMemcpy(scores,d_scores,n*sizeof(int),cudaMemcpyDeviceToHost);


            

            // wait for the rest of results
            for(int i = 0; i < n; i++) {
          
                if(scores[i] > best_score) {
                    best_score = scores[i];
                    best_score_index = i;
                }
              //  printf("received score %i from %i\n", scores[i], i);
            }

            put_symbol(board, current_symbol, moves[best_score_index]);

            print_board(board);

            for(int i = 0; i < n; i++) {
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
                printf(COLOR_RESET"enter move accordingly ");
                scanf("%d",&playMove);
                    put_symbol(board, current_symbol, moves[playMove-1]);

                    print_board(board);

            }
            current_symbol = 1 - current_symbol;

        }

        

        if(result==0)
        {
            printf(COLOR_GREEN"\t\tCPU Wins\n");
            exit(0);
        }
    else
        if(result==1)
            {
                printf(COLOR_GREEN"\t\tPlayer Wins\n");
                exit(0);

            }




        }

