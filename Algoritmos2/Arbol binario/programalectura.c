#include<stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct NODOARBOL
{ 
    int dato;
    struct NODOARBOL *izq;
    struct NODOARBOL *der;
} NODOARBOL;

typedef struct{
    char ClaveUDA[10];
    char NombreUDA[60];
    char TipoUDA[80];
    char AreaUDA[80];
    unsigned int NumCred;
    unsigned int Hrs;

} REGISTROS;


typedef enum{PREDECESOR, SUCESOR}SUSTITUTO;

NODOARBOL* CrearNodoArbol(int dato);
void InsertarNodoEnArbol(int dato,NODOARBOL **Arbol,char listado[72][60]);
void MostrarArbol(NODOARBOL *Arbol,char listado[72][60]);
void EliminarArbol(NODOARBOL **Arbol);
unsigned int Max(unsigned int A, unsigned int B);
unsigned int AlturaDelArbol(NODOARBOL *Arbol);
unsigned int NumeroDeNodos(NODOARBOL *Arbol);
unsigned int NumeroDeHojas(NODOARBOL *Arbol);
unsigned int NumeroDeRamas(NODOARBOL *Arbol);
NODOARBOL* BuscarDato(char *clave, NODOARBOL *Arbol, char listado[][60]);
REGISTROS* GetRegistro(int indice);
int main()
{
    NODOARBOL* Abb = NULL;
    NODOARBOL* NodoAux = NULL;
    FILE* archivo;
    unsigned int op = 0;
    char listado[72][60], aux[60];
    
    do
    {
        printf("\n--Programa de Busqueda--");
        printf("\n1. Busqueda por Clave de UDA.");
        printf("\n2. Busqueda por Nombre de UDA.");
        printf("\n3. Salir.");
        printf("\nSu opcion:  ");
        scanf("%u",&op);
        if((op<=0)||(op>=4))
            printf("\nOpcion no validaaa!!!");
    }while((op<=0)||(op>=4));

    archivo = fopen("D:/Escuela/4to Semestre/Estructuras y Algoritmos/listadoNoOrden.txt","r");
    
    if(archivo!=NULL)
    {
        printf("\nLeyendo archivo..");
        for(int i=0; i<72; i++)
        {
            fgets(&listado[i][0],60,archivo);
            fgets(aux,60,archivo);
            fgets(aux,60,archivo);
            fgets(aux,60,archivo);
            fgets(aux,60,archivo);
            fgets(aux,60,archivo);
        }
        fclose(archivo);
        printf("\nArchivo leido!\n");
        //printf("\nListado:\n");
        for(int i=0; i<72; i++)
            InsertarNodoEnArbol(i, &Abb, listado);
        //MostrarArbol(Abb,listado);

            //printf("\n\nAltura del arbol: %i",AlturaDelArbol(Abb));
            //printf("\nHojas del arbol: %i",NumeroDeHojas(Abb));
            //printf("\nNodos del arbol: %i",NumeroDeNodos(Abb));
            //printf("\nRamas del arbol: %i\n",NumeroDeRamas(Abb));
        }
        else
            printf("No fue posible abrir el archivo");


    if(op==1)
    {
        printf("\n busqueda por clave de UDA: ");
        //fgets(aux);
        //fgets(aux);
        printf("\nClave a buscar: %s", aux);
        NodoAux = BuscarDato(aux,Abb, listado);
        if(!NodoAux)
            printf("\nClave encontrada, en el indice %i",NodoAux->dato);
        else
            printf("\nClave no encontrada");
    }
    if(op==2)
    {
        //printf("\nDato pene",);
    }

    if(op==3)
    {
        printf("\nPrograma terminado correctamente.");
    }

    return 0;
}


void InsertarNodoEnArbol(int dato,NODOARBOL **Arbol,char listado[72][60])
{ 
    NODOARBOL *NewNodo;
    //NewNodo=CrearNodoArbol(dato);
    static unsigned int flag=1;
    //Verificar si el arbol esta vacio
    if(*Arbol==NULL){
        NewNodo=CrearNodoArbol(dato);
        *Arbol=NewNodo;
        //*Arbol=se refiere al apuntador que llega
        //**Arbol=apunta a lo que apunta el apuntador que llega
    }else{
        //El arbol contiene el nodo raiz
        if( strcmp(&listado[dato][0],&listado[(*Arbol)->dato][0])==-1 ){
            InsertarNodoEnArbol(dato,&((*Arbol)->izq),listado);
        }
        else{
            InsertarNodoEnArbol(dato,&((*Arbol)->der),listado);
        }        
    }
}
NODOARBOL* CrearNodoArbol(int dato)
{
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

void MostrarArbol(NODOARBOL *Arbol, char listado[72][60])
{
  NODOARBOL *aux;
  if(Arbol==NULL)
    printf("\nArbol vacio\n");
  else
  {
    aux=Arbol;
    if(aux->izq!=NULL)
    {
      aux=aux->izq;
      MostrarArbol(aux,listado);
    }
    aux=Arbol;
    printf("[%i]: %s",aux->dato, &listado[aux->dato][0]);
    if(aux->der!=NULL)
    {
      aux=aux->der;
      MostrarArbol(aux,listado);
    }
  }
}

void EliminarArbol(NODOARBOL **Arbol){
    if(*Arbol==NULL)
        return;
    else{
        if(((*Arbol)->izq)!=NULL)
            EliminarArbol(&((*Arbol)->izq));
        if(((*Arbol)->der)!=NULL)
            EliminarArbol(&((*Arbol)->der));
        //printf("\nSe elimino el nodo: %s",(*Arbol)->dato);
        free(*Arbol);
        *Arbol=NULL;
    }
}

unsigned int Max(unsigned int A, unsigned int B)
{
  if(A < B)
    return B;
  else
    return A;
}

unsigned int AlturaDelArbol(NODOARBOL *Arbol)
{
    if(Arbol == NULL)
        return 0;
    else
        return(1+ Max(AlturaDelArbol(Arbol->izq), AlturaDelArbol(Arbol->der)));
}

unsigned int NumeroDeNodos(NODOARBOL *Arbol){
    if(Arbol==NULL)
        return 0;
    else{
        return 1+NumeroDeNodos(Arbol->izq)+NumeroDeNodos(Arbol->der);
    }
}

unsigned int NumeroDeHojas(NODOARBOL *Arbol){
    if(Arbol==NULL)
        return 0;
    if((Arbol->izq==NULL)&&(Arbol->der==NULL))
        return 1;
    else
        return NumeroDeHojas(Arbol->izq)+NumeroDeHojas(Arbol->der);
}

unsigned int NumeroDeRamas(NODOARBOL *Arbol){
    if(Arbol==NULL)
        return 0;
    else
        return NumeroDeNodos(Arbol)-NumeroDeHojas(Arbol)-1;
}

NODOARBOL* BuscarDato(char *clave, NODOARBOL *Arbol, char listado[][60])
{
    if(Arbol!=NULL)
    {
        if(strcmp(clave, &listado[(*Arbol)->dato][0]) == 0)
            return Arbol;
        else
        {
            if(strcmp(clave, &listado[(*Arbol)->dato][0]) == -1)
                return BuscarDato(clave,Arbol->izq, listado);
            else
                return BuscarDato(clave,Arbol->der, listado);
        }
    }
    return NULL; 
}


REGISTROS* GetRegistro(int indice, char[] filename)
{
    REGISTROS* ptr;
    ptr = (REGISTROS*)malloc(sizeof(REGISTROS));
    if (ptr == NULL) 
    {
        printf("no se pudo xdxd");
        return(0);
        exit(0);
    }
}