/*Programa No.2
    Implementacion de un Arbol Binario de Busqueda de informacion
    por distintas claves.
        //printf("\nstrcmp(uno,dos)= %i", strcmp("uno","dos"));
        //printf("\nstrcmp(hoja,bueno)= %i", strcmp("azul","bueno"));
        //printf("\nstrcmp(uno,uno)= %i", strcmp("uno","uno"));
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct NODOARBOL{ 
    char *dato;
    struct NODOARBOL *izq;
    struct NODOARBOL *der;
}NODOARBOL;

typedef enum{PREDECESOR, SUCESOR}SUSTITUTO;

typedef struct NODOFILA{
    NODOARBOL *dato;
    struct NODOFILA *next;
}NODOFILA;
NODOARBOL* CrearNodoArbol(char* dato);
void InsertarNodoEnArbol(char* dato,NODOARBOL **Arbol);
void MostrarArbol(NODOARBOL *Arbol);
void EliminarArbol(NODOARBOL **Arbol);
unsigned int AlturaDelArbol(NODOARBOL *Arbol);
unsigned int Max(unsigned int A, unsigned int B);
unsigned int NumerosdeNodos(NODOARBOL *Arbol);
unsigned int NumeroDeHojas(NODOARBOL *Arbol);
unsigned int NumeroDeRamas(NODOARBOL *Arbol);
/*

void PosOrden(NODOARBOL *Arbol);
void InOrden(NODOARBOL *Arbol);
void PreOrden(NODOARBOL *Arbol);

NODOFILA* CrearNodoFila(NODOARBOL *nodo);
void Enfilar(NODOFILA **fila, NODOARBOL *nodo); 
void Desenfilar(NODOFILA **fila);
NODOARBOL* FrenteDeLaFila(NODOFILA *fila);
//unsigned char FilaVacia(NODOFILA *fila);
void MostrarFila(NODOFILA *fila);
void RecorridoporNivel(NODOARBOL *Arbol);

NODOARBOL * BuscarDato(int dato, NODOARBOL *Arbol);
NODOARBOL *BuscarMenor(NODOARBOL *Arbol);
NODOARBOL *BuscarMayor(NODOARBOL *Arbol);
void EliminarNodo(int datox, NODOARBOL **Arbol,SUSTITUTO tipo);
*/
int main(){
    FILE* Archivo;
    NODOARBOL* Abb=NULL;
    char listado[72][60];
    char aux [60];

    char NombredelArchivo[]="listadoNoOrden.txt";
    Archivo=fopen(NombredelArchivo,"r");
    unsigned int op = 0;
    do 
    {
        printf("\n0. Programa de búsqueda");
        printf("\n1. Busqueda de clave");
        printf("\n2. Busqueda de nombre de uda");
        printf("\n3. Salir.");
        printf("\nSu eleccion:");
        scanf("%u",&op);
        if(!(op>0 && op<4))
        {
            printf("\nSelección no válida");
        }

    }while(!(op>0 && op<4));

    if (op == 3)
    {
        printf("cerrando");
        exit(0);
    }
    if(Archivo!=NULL)
    {
        printf("\nLeyendo Archivo: (%s)",NombredelArchivo);
        if (op == 1)
            for(int i=0; i<72; i++)
            {
                fgets(&listado[i][0], 60, Archivo);
                fgets(aux,60,Archivo);
                fgets(aux,60,Archivo);
                fgets(aux,60,Archivo);
                fgets(aux,60,Archivo);
                fgets(aux,60,Archivo);
            }
        if (op == 2)
        {
            printf("\nNo está implementado");
            fclose(Archivo);
            return 0;
        }
        fclose(Archivo);
        printf("\nArchivo leido con exito.\n");
        //for(int i=0; i<71; i++)
        //    printf("%i. %s",i+1,&listado[i][0]);
        for(int i=0; i<72; i++)
            InsertarNodoEnArbol(&listado[i][0],&Abb);
        printf("\nABB creado:\n");
        MostrarArbol(Abb);
        printf("\nInfo del arbol");
        printf("\nAltura del arbol %u", AlturaDelArbol(Abb));
        printf("\n Numero de nodos: %u", NumerosdeNodos(Abb));
        printf("\n Numero de hojas: %u", NumeroDeHojas(Abb));
        printf("\n Numero de ramas: %u", NumeroDeRamas(Abb));
        EliminarArbol(&Abb);
        MostrarArbol(Abb);
    }else   
        printf("2");
    return 0;
}

unsigned int AlturaDelArbol(NODOARBOL *Arbol)
{
    if(Arbol == NULL)
        return 0;
    else
        return(1+ Max(AlturaDelArbol(Arbol->izq), AlturaDelArbol(Arbol->der)));
}
unsigned int Max(unsigned int A, unsigned int B)
{
    if(A > B)
        return A;
    else
        return B;
}
unsigned int NumerosdeNodos(NODOARBOL *Arbol)
{
    if(Arbol == NULL)
        return 0;
    else
        return 1 + NumerosdeNodos(Arbol->izq) + NumerosdeNodos(Arbol->der);
}
unsigned int NumeroDeHojas(NODOARBOL *Arbol)
{
    if(Arbol == NULL)
        return 0;
    if((Arbol->izq == NULL) && (Arbol->der == NULL))
        return 1;
    else
        return NumeroDeHojas(Arbol->izq) + NumeroDeHojas(Arbol->der);
}
unsigned int NumeroDeRamas(NODOARBOL *Arbol)
{
    if(Arbol == NULL)
        return 0;
    else
        return NumerosdeNodos(Arbol) - NumeroDeHojas(Arbol) - 1;
}

void EliminarArbol(NODOARBOL **Arbol)
{
    if(*Arbol == NULL)
        return;
    else
    {
        if(((*Arbol)->izq) != NULL)
            EliminarArbol(&((*Arbol)->izq));
        if(((*Arbol)->der) != NULL)
            EliminarArbol(&((*Arbol)->der));
        free(*Arbol);
        *Arbol = NULL;
    }
}
void MostrarArbol(NODOARBOL *Arbol){
    NODOARBOL *aux;
    if(Arbol==NULL)
        printf("\nArbol vacio\n");
    else{
        aux=Arbol;
        if(aux->izq!=NULL)
        {
            aux=aux->izq;
            MostrarArbol(aux);
        }
        aux=Arbol;
        printf("%s",aux->dato);
        if(aux->der!=NULL)
        {
            aux=aux->der;
            MostrarArbol(aux);
        }
    }
}
NODOARBOL* CrearNodoArbol(char* dato){
    NODOARBOL *ptr;
    ptr=(NODOARBOL*)malloc(sizeof(NODOARBOL));
    if(ptr==NULL){
        printf("ERROR:No se genero el espacio de memoria\n");
        exit(0);
    }
    ptr->dato=dato;
    ptr->izq=NULL;
    ptr->der=NULL;
    return ptr;
}
void InsertarNodoEnArbol(char* dato,NODOARBOL **Arbol){
    NODOARBOL *NewNodo;
    static unsigned int flag=1;
    //Verificar si el arbol esta vacio
    if(*Arbol==NULL){
        NewNodo=CrearNodoArbol(dato);
        *Arbol=NewNodo;
    }else{
        if(strcmp(dato,(*Arbol)->dato)==-1)
            InsertarNodoEnArbol(dato,&((*Arbol)->izq));
        if(strcmp(dato,(*Arbol)->dato)==1)
            InsertarNodoEnArbol(dato,&((*Arbol)->der));      
    }
}