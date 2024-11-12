#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//constants
const int POLY_MAX_C = 10;

//globals
double coeficients[POLY_MAX_C];
unsigned n_coefficients;
double lim_a;
double lim_b;
unsigned n_segments;

void print_help()
{
    printf("usage: $ polyint a b d C0 [ C1 | C2 | ...]\n");
    printf("a - Lower bound\n");
    printf("b - Upper bound\n");
    printf("d - segments between a and b interval\n");
    printf("Cn - the coeficients, add as many as needed.\n");
}

void error_message(char* message, int errorcode)
{
    printf("error code %i\n %s\n", errorcode, message);
    exit(errorcode);
}

void warning_message(char* message)
{
    printf("%s", message);
}

void swap_bounds(double *a, double *b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}

double poly_eval (const double* coefficients, unsigned n_coefficients, double x)
{
    double y = 0.0;
    double exp = 1.0;
    for (unsigned i = 0; i < n_coefficients; i++)
    {
        y += coefficients[i]*exp;
        exp*=x;
    }
    return y;
}

double eval_range(double a, double b)
{
    double sum = 0;
    double step = (b - a) / (double) n_segments;
    double y0  = poly_eval(coeficients, n_coefficients, a), y1;
    double rect_height = 0;
    double tri_height = 0;

    for (unsigned i = 1; i <= n_segments; i++)
    {
        y1 = poly_eval(coeficients, n_coefficients, a + i * step);

        rect_height = (y0 <= y1) ? y1 : y0;

       tri_height = (y0 < y1) ? y1 - y0 : y0 - y1;

        sum += step * rect_height; //rectangulo
        sum += 0.5 * step * tri_height;

        y0 = y1;
    }
    return sum;
}

int main(int argc, char** argv)
{
    //mpi
    MPI_Init(NULL, NULL);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 5) print_help();
    n_coefficients = argc - 4;
    lim_a = atof(argv[1]);
    lim_b = atof(argv[2]);
    n_segments = atoi(argv[3]);

    for (unsigned i = 0;  i <n_coefficients; i ++)
    {
        coeficients[i] = atof(argv[4+i]);
    }

    if (n_coefficients > POLY_MAX_C) error_message("Too many coefficients, max is 10.", 3);
    if (lim_a > lim_b)
    {
        swap_bounds(&lim_a, &lim_b);
        warning_message("lower bound is higher than upper bound, they have been swapped.");
    }
    if (n_segments == 0) error_message("segments must be different from 0", 4);

    double s = (lim_b - lim_a)/size;
    double a = lim_a + rank * s;
    double b = a + s;

    double sum = eval_range(a, b);
    double total_sum = 0;

    MPI_Reduce(&sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Result = %g\n", total_sum);
    }



    //printf("Result = %g\n", eval_range(a, b));
    MPI_Finalize();
}
