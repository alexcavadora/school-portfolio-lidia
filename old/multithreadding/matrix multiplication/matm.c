#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// *** Memoria compartida ***
unsigned g_mcols = 0;
unsigned g_mrows = 0;
unsigned* g_matrix = NULL;

unsigned g_vcols = 0;
unsigned g_vrows = 0;
unsigned* g_vector = NULL;

unsigned* g_vresult = NULL;
// *** ------------------ ***

void print_help(void)
{
    printf("usage matm <matrix> <vector> \n");
    exit(1);

}

void error_msg(const char* msg, unsigned err)
{
    printf("Error: %s\n", msg);
    exit(err);
}

unsigned* load_matrix(char* filename, unsigned* num_of_rows, unsigned* num_of_cols)
{
    unsigned* matrix = NULL;
    FILE* file_descriptor = fopen(filename, "rt");
    if (file_descriptor == NULL) return NULL;

    char buffer[BUFSIZ]; //buffer para leer una sola línea
    unsigned rows, cols; //variables para almacenar las dimensiones

    if(fgets(buffer, BUFSIZ, file_descriptor) == NULL) goto fail;
    rows = atoi(buffer);

    if(fgets(buffer, BUFSIZ, file_descriptor) == NULL) goto fail;
    cols = atoi(buffer);

    unsigned size = rows*cols;
    if (size == 0) goto fail;
    matrix = malloc(size*sizeof(unsigned));
    if (matrix == NULL) goto fail;


    for(unsigned i = 0; i < size; i++)
    {
        if(fgets(buffer, BUFSIZ, file_descriptor) == NULL) goto fail;
        matrix[i] = atoi(buffer);
    }
    fclose(file_descriptor);

    //returns
    *num_of_cols = cols;
    *num_of_rows = rows;
    return matrix;
fail:
    if(matrix != NULL) free(matrix);
    fclose(file_descriptor);
    return NULL;
}

void print_matrix(unsigned* matrix, unsigned rows, unsigned cols)
{
    printf("rows = %i\n", rows);
    printf("cols = %i\n", cols);
    for (unsigned r = 0; r < rows; r++)
    {
        for(unsigned c = 0; c < cols; c++)
            printf("%i ", matrix[r*cols+c]);
        printf("\n");
    }
}

void* row_times_column(void* args)
{
    unsigned* row = (unsigned*) args;
    unsigned index = (unsigned)(row - g_matrix) / g_mcols;

    unsigned y = 0;
    for (unsigned i = 0; i < g_mcols; i++)
    {
        y += row[i] * g_vector[i];

    }
    g_vresult[index] = y;
    return NULL;
}

int main(int argc, char** argv)
{
    if (argc !=3) print_help();
    char* matrix_file = argv[1];
    char* vector_file = argv[2];

    g_matrix = load_matrix(matrix_file, &g_mrows, &g_mcols);
    if (g_matrix == NULL) error_msg("Wrong matrix", 2);

    g_vector = load_matrix(vector_file, &g_vrows, &g_vcols);
    if (g_vector == NULL) error_msg("Wrong vector", 3);

    if(g_mcols != g_vrows) error_msg("Uncompatible dimensions", 4);

    g_vresult = malloc(g_mrows * sizeof(unsigned));

    pthread_t* threadls = malloc(g_mrows * sizeof(pthread_t));
    if (threadls == NULL) error_msg("Threads Alloc fail", 5);

    for(unsigned i = 0; i<g_mrows; i++)
        pthread_create(&threadls[i], NULL, row_times_column, (void*) &g_matrix[i*g_mcols]);

    for(unsigned i = 0; i < g_mrows; i++)
        pthread_join(threadls[i], NULL);

    printf("Result: \n");
    print_matrix(g_vresult, g_mrows, g_vcols);
    if (g_matrix != NULL) free(g_matrix);
    if (g_vector != NULL) free(g_vector);
    if (g_vresult != NULL) free(g_vresult);

    return 0;
}


//[ ][ ][4][ ][ ]
// 0  1  2  3  4
//       ^
//       |
//       x
// unsigned* y = &x
// y = 2, el índice en memoria
// *y = 4, el valor en memoria en el índice 2
