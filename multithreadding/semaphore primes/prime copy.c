#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void print_help() {
  printf("usage: semaphore <nthreads> <batch> <nprimes>\n");
  printf("nthreads:  Threads amount number.\n");
  printf("batch:      Amount of numbers to try per batch.\n");
  printf("nprimes:    Total amount of primes.\n");
}

void error_msg(const char *msg) {
  fprintf(stderr, "ERROR: %s\n", msg);
  exit(2);
}

///////////////////// REGION CRITICA ///////////////////////
pthread_mutex_t cr_mutex = PTHREAD_MUTEX_INITIALIZER;
long long *cr_prime_list = NULL; // lista de primos
long long cr_i = 0;          // rango inferior en hilos
long long cr_k = 0;          // rango mayor en hilos
long long cr_next = 2;           // siguiente numero a probar
long long cr_lmax = 2;          // Numero maximo probado para gennerar la lista
//-------------------------------------------------

long long g_nprimes = 0;  // primos a calcular
unsigned  g_batch = 0;    // batch por hilo
unsigned  g_nthreads = 0; // numero de hilos

// BARRIER
pthread_mutex_t cr_bmut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cr_cond = PTHREAD_COND_INITIALIZER;
unsigned cr_wait = 0; // hilos esperando en barrer

int is_prime_force(long long num, long long start, long long end) {
  long long div = (start & 1) ? start : start + 1;
  while(div <= end)
  {
    if(num%div==0) return 0;
    div = div+2;
  }
  return 1;
}

int is_prime(long long n)
{
    if (n == 0)
      return 0;
    if (n == 1)
      return 0;
    if (n == 2)
      return 1;
    if (n == 3)
      return 1;
    if (n == 4)
      return 0;
    if (n == 5)
      return 1;

    if (n % 2 == 0)
      return 0;
    if (n % 5 == 0)
      return 0;
    long long ns = 1 + (long long) floor(sqrt((double) n));
    for (unsigned i = 0; i < cr_i; i++)
    {
        if(cr_prime_list[i] > ns) continue;
        if(n % cr_prime_list [i] == 0) return 0;
    }

    if (ns > cr_lmax) return is_prime_force(n, cr_lmax, ns);
    return 1;
}

void *prime_thread(void *arg) {
  int id = *((int *)arg);
  free(arg);
  long long int *list = malloc(sizeof(long long int) * g_batch / 2);
  int cont = 1;
  long long num = 0; // numero a probar
  long long end = 0; // límite del batch
  unsigned n = 0;
  while (cont) {
    //--------------------Critical Region ------------------------
    pthread_mutex_lock(&cr_mutex);

    if (cr_i < g_nprimes)
    {
      num = cr_next;
      cr_next += g_batch;
      end = cr_next;
    } else
      cont = 0;
    pthread_mutex_unlock(&cr_mutex);
    //--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    n = 0;
    while (num < end)
    {
        if (is_prime(num))
        {
            list[n] = num;
            n++;
        }
        num++;
    }
    //--- --- --- Contribución --- --- ---
    pthread_mutex_lock(&cr_mutex);
        for (int i = 0; i < n; i++)
            cr_prime_list[cr_k++] = list[i];

    pthread_mutex_unlock(&cr_mutex);

    //--- --- --- Barrier --- --- ---
    pthread_mutex_lock(&cr_bmut);
        cr_wait++;
        if (cr_wait == g_nthreads) {
        cr_wait = 0;
        cr_i = cr_k;
        pthread_cond_broadcast(&cr_cond);
        } else
        while (pthread_cond_wait(&cr_cond, &cr_bmut) != 0)
    pthread_mutex_unlock(&cr_bmut);
  }
  free(list);
  return NULL;
}

int *intp(int n) {
  int *p = malloc(sizeof(int));
  if (p != NULL)
    *p = n;
  return p;
}

int main(int argc, char **argv) {
  if (argc != 4)
    print_help();
  g_nthreads = atoi(argv[1]);
  g_batch = atoi(argv[2]);
  g_nprimes = atol(argv[3]);

  if (g_nthreads > 200 || g_nthreads < 1)
    error_msg("Invalid number of threads.");
  if (g_batch == 0)
    error_msg("Invalid number of batches.");
  if (g_nprimes == 0)
    error_msg("Invalid number of primes.");

  cr_prime_list = malloc(g_nprimes * sizeof(long long));
  if (cr_prime_list == NULL)
    error_msg("Couldn't initiate prime list, free some memory.");

  pthread_t *tlist = malloc(g_nthreads * sizeof(pthread_t));
  if (tlist == NULL)
    error_msg("Couldn't initiate threads, free some memory.");

  for (int i = 0; i < g_nthreads; i++) {
    pthread_create(&tlist[i], NULL, prime_thread, intp(i));
  }

  for (int i = 0; i < g_nthreads; i++) {
    pthread_join(tlist[i], NULL);
  }

  free(tlist);
}
