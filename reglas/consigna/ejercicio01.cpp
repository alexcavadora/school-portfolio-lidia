#include <iostream>

using namespace std;
int main()
{
    int size = 10;
    float x[size];
    for (int i = 0; i < size; i++)
    {
        cout << "Enter a number: ";
        cin >> x[i];
        if (x[i] >= 400 && x[i] <= 500)
            cout << "En el rango."<< endl;
        if ((int) x[i] == x[i] && x[i] != 0)
            cout << "Entero."<< endl;
        if ((int) x[i] % 13 == 0)
            cout << "Múltiplo de 13."<< endl;
    }

    for (int i = 0; i < size; i++)
    {
        cout << "Number in index: "<< i << " = "<< x[i] << "\n";
    }
    return 0;
}
