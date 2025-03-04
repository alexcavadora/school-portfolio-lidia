#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <mutex>
#include <cmath>

// --- CRITICAL REGION -----------------------------------------
std::mutex cr_mutex;
uint64_t   g_start     = 0;
uint64_t   cr_next      = 0; // siguiente número no evaluado
unsigned   cr_count     = 0; // contador de cr_primelist
unsigned   g_nprimes   = 0;
unsigned   g_batch = 0;
uint64_t*  cr_primelist = nullptr;
//--------------------------------------------------------------

bool is_prime( uint64_t num ) {

   if ( num == 0 ) return false;
   if ( num == 1 ) return false;
   if ( num == 2 ) return true;
   if ( num == 3 ) return true;
   if ( num == 4 ) return false;
   if ( num == 5 ) return true;

   if ( num % 2 == 0 ) return false;
   if ( num % 5 == 0 ) return false;

   uint64_t nmax = 1 + (uint64_t)floor(std::sqrt((double)num));

   unsigned step[4] = {4, 2, 2, 2};
   unsigned c = 0;

   uint64_t div = 3;
   while ( div < nmax ) {
      if ( num % div == 0 ) return false;
      div = div + step[c];
      c = ( c == 3 ) ? 0 : c + 1;
      //c ++;
      //if ( c == 4 ) c = 0;
   }

   return true;
}

void prime_thread( void ) {
   uint64_t  num = 0;
   uint64_t  end = 0;
   unsigned  n   = 0;

   uint64_t* list = new uint64_t[g_batch/2];

   while ( true )
   {
       bool stop = false;
        //--------------------------------------------
        cr_mutex.lock();

        if ( n > 0 )
        {
            n = (cr_count + n > g_nprimes) ? g_nprimes - cr_count : n;
            for ( unsigned i=0; i<n; i++ )
            {
                cr_primelist[cr_count] = list[i];
                cr_count ++;
            }
        }

        if(cr_count < g_nprimes)
        {
            num = cr_next;
            end = cr_next + g_batch; // < end
            cr_next += g_batch;
        }
        else
            stop = true;
        cr_mutex.unlock();
      //--------------------------------------------
      if (stop == true)
        break;
      n = 0;
      while ( num < end )
      {
         if ( is_prime(num) )
         {
            list[n] = num;
            n ++;
         }
         num ++;
      }
   }

   delete [] list;
}

void print_help() {
   std::cout << "usage: prime <threads> <batch> <n-primes> <start>\n";
   std::cout << "threads   Número de hilos.\n";
   std::cout << "batch     Números a probar por hilo.\n";
   std::cout << "n-primes  Total de números primos a encontrar.\n";
   std::cout << "start     Número inicial.\n";
}

void error_msg( const char* msg, int err ) {
   std::cout << "ERROR: " << msg << "\n";
   std::exit(err);
}

int main( int argc, char** argv ) {

   if ( argc != 5 ) {
      print_help();
      return 1;
   }

    unsigned nthreads  = std::stoul(argv[1]);
    g_batch = std::stoul(argv[2]);
    g_nprimes   = std::stoul(argv[3]);
    g_start     = std::stoul(argv[4]);
    cr_next = g_start;

    cr_primelist = new uint64_t[g_nprimes];
    if ( cr_primelist == nullptr ) error_msg("alloc failed",5);

    if ( nthreads  == 0  ) error_msg("zero threads", 2);
    if ( nthreads  >  32 ) error_msg("too many threads", 3);
    if ( g_batch == 0  ) error_msg("zero batch", 4);
    if ( g_nprimes   == 0  ) return 0;



    std::thread* tlist = new std::thread[nthreads];
    for (unsigned i = 0; i < nthreads; i++)
    {
        tlist[i] = std::thread(prime_thread);
    }
    for (unsigned i = 0; i < nthreads; i++)
    {
        tlist[i].join();
    }
    std::cout << "Primes: " << std::endl;
    for (unsigned i = 0; i < nthreads; i++)
    {
        std::cout << cr_primelist[i] << std::endl;
    }
    delete [] cr_primelist;
    return 0;
}
