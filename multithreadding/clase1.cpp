#include<iostream>
#include<thread>
#include<mutex> //variable atómica que puedo modificar o no, pero no se puede quedar a medias.

//región crítica------
std::mutex cr_mutex;
//--------------------
void hello(int n)
{
    std::cout << "Hello, world!, desde el hilo "<< n << "\n";
    return;
}

int main(int argc, char** argv)
{
     std::cout << "Inicia main \n";

    if (argc!=2)
    {
        std::cout << "usage hello <threads>" << "\n";
        return 1;
    }
    int nthreads = std::stoi(argv[1]);
    //std::cout << "No. threads = " << nthreads << "\n";

    if (nthreads > 200)
    {
        std::cout << "too many threads..." << "\n";
        return 1;
    }
    std::thread* list = new std::thread[200];
    for (int i = 0; i < nthreads; i++)
    {
        list[i] = std::thread(hello, i+1);
    }

    for (int i = 0; i < nthreads; i++)
    {

        ;
    }

    // std::thread t1(hello, 1);
    // std::thread t2(hello, 2);
    // std::thread t3(hello, 3);
    // t1.join();
    // t2.join();
    // t3.join();
    std::cout << "Finaliza main\n";
	return 0;
}
