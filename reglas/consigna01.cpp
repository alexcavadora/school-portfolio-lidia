#include <iostream>
using namespace std;
int n_inscripciones;
int n_creditos;
float cps;
int main()
{
    cout << "Inscripciones: ";
    cin >> n_inscripciones;
    cout << "Créditos: ";
    cin >> n_creditos;
    cps = (float)n_creditos/(float)n_inscripciones;
}
