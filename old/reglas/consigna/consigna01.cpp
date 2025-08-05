#include <iostream>
#include <cmath>  

void mostrarCategoriaDesempeno(int inscripciones, int creditos) 
{
    if (inscripciones <= 0 || creditos < 0) 
    { 
        std::cout << "Datos inválidos. El número de inscripciones debe ser mayor que 0 y los créditos no pueden ser negativos." << std::endl;
        return;
    }

    double PCS = static_cast<double>(creditos) / inscripciones;
    std::cout << "Promedio de Créditos por Semestre (PCS): " << PCS << std::endl;

    if (PCS > 33) {
        std::cout << "Categoría: Desempeño_Alto" << std::endl;
    } else if (PCS >= 28 && PCS <= 33) {
        std::cout << "Categoría: Desempeño_Normal" << std::endl;
    } else if (PCS >= 23 && PCS < 28) {
        std::cout << "Categoría: Rezago_Moderado" << std::endl;
    } else if (PCS < 23) {
        std::cout << "Categoría: Rezago_Extremo" << std::endl;
    }

    int semestresRestantes = std::ceil((270 - creditos) / PCS);
    if (semestresRestantes > 0) {
        std::cout << "Semestres restantes para concluir el programa: " << semestresRestantes << std::endl;
    } else {
        std::cout << "El alumno ya ha acumulado suficientes créditos para concluir el programa." << std::endl;
    }
}

int main() 
{
    int inscripciones, creditos;

    std::cout << "Introduce el número de inscripciones (semestres cursados): ";
    std::cin >> inscripciones;
    std::cout << "Introduce el número de créditos acumulados: ";
    std::cin >> creditos;

    mostrarCategoriaDesempeno(inscripciones, creditos);

    return 0;
}
