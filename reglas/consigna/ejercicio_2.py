#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 01:31:43 2023
@date: 10 nov 2023
@subject: Data Analysis Fundamentals
@author: Alejandro Alonso Sánchez
@description: Puzzle deslizante 4x4, se genera de forma aleatoria un juego con posibilidad
		      de victoria, debes mover el hueco, en este caso representado por un 0 para 
		      ordenar los números de manera ascendente.
"""

import random as rn



def setup():
    """
    Inicializa el tablero aleatoriamente, generando juegos hasta hallar uno resolvible  
    Algo curioso es que técnicamente podría generar una matriz resuelta ya que el módulo de 0 en base 2 es 0.
    Se podría remover al comprobarlo con que no esté resuelta pero me gusta.
    
    Returns
    -------
    list
        Matriz aleatoria resolvible.

    """
    n = 1
    while n % 2 == 1:
        n = 0
        
        numbers = [i+1 for i in range(15)] #se genera una lista con los números del 1 al 15
        rn.shuffle(numbers) # se randomiza la lista (2D), si se comenta se puede debuggear la victoria.
        for i in range(len(numbers)):
            for j in range(i):
                if numbers[j] > numbers [i]: #se comprueba que sea posible resolver, de lo contrario genera una distinta hasta hallar una.
                    n+=1
    numbers.append(0) #al final del todo, se agrega el 0 y se convierte en matriz 4x4
    return [numbers[:4], numbers[4:8], numbers[8:12], numbers[12:16]]


def printMatrix(matrix):
    """
    Imprime la matriz, en forma de tablero, usando formateo adecuado dependiendo
    de la longitud de cada número en las celdas.
    
    Parameters
    ----------
    matrix : list
        Contiene en listas los renglones de entrada.

    Returns
    -------
    Nada.

    """
    for i in matrix: # se imprime la matriz, se usa un operador ternario para imprimir los números de 2 dígitos sin espacio y los de 1 con espacio
        print('╔═══╗╔═══╗╔═══╗╔═══╗')
        print(f'║ {i[0]}{(" ","")[i[0] > 9]}║║ {i[1]}{(" ","")[i[1] > 9]}║║ {i[2]}{(" ","")[i[2] > 9]}║║ {i[3]}{(" ","")[i[3] > 9]}║')
        print('╚═══╝╚═══╝╚═══╝╚═══╝')
    print('\n')


def solved(matrix):
    """
    Chequeo de solución, ignora la posición del hueco

    Parameters
    ----------
    matrix : list
        Contiene en listas los renglones de entrada.

    Returns
    -------
    bool: True: está resuelto.
          False: no está resuelto.

    """
    prev = 0
    for i in matrix: #se comprueba que esté resuelto, al comprobar que los números siguan una secuencia ascendente
        for j in i:
            if j == 0:
                continue
            if j != prev+1:
                return False
            else:
                prev = j
    return True

#soluciona mediante fuerza bruta
def solve(matrix):
    """
    Resuelve el juego al remplazar todos los números con números en orden.

    Parameters
    ----------
    matrix : TYPE
        Contiene todos los números en forma de matriz.

    Returns
    -------
    None.

    """
    n = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = n
            n+=1
    return 

def search(value, matrix):
    """
    Busca un número y regresa sus coordenadas, se usa para ubicar al 0.

    Parameters
    ----------
    value : INT
        Valor a buscar.
    matrix : list
        Contiene todos los números en forma de matriz.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == value:
                return j, i
    return 'No se encontró ni maiz...'
     
#intercambia 2 casillas, la del 0 y otra   
def swap_zero(matrix, x1, y1, x2, y2):
    matrix[y1][x1] = matrix[y2][x2]
    matrix[y2][x2] = 0
    return

#checa si el movimiento es legal en base a su posición actual y la tecla sugerida
def legal(c, x, y):
    if (c == 'w' and  y > 0) or (c == 'a' and  x > 0) or (c == 's' and  y < 3) or (c == 'd' and  x < 3) or (c == 'solve'):
        return True
    print('No se puede realizar ese movimiento.')
    return False

#logica de movimiento, si es legal, lo hace, si se escribe solve se soluciona automáticamente (reemplazo)
def move(c, matrix):
    x, y = search(0, matrix)
    if legal(c, x, y):
        if c == 'solve':
            solve(matrix)
            return
        
        elif c == 'w':
            swap_zero(matrix, x, y, x, y-1)
            y -= 1
        
        elif c == 'a':
            swap_zero(matrix, x, y, x-1, y)
            x -= 1
            
        elif c == 's' :
            swap_zero(matrix, x, y, x, y+1)
            y += 1
        
        elif c == 'd':
            swap_zero(matrix, x, y, x+1, y)
            x += 1
    return
        
#repite el juego tantas veces quieras
def game_deslizante():
    print('Instrucciones: puedes controlar el número 0, que representa el hueco, mediante las teclas w, a, s, d, que indican arriba, izquierda, abajo y derecha respectivamente. Ordena los números del 1 al 15 dejando el 0 en la última casilla.')
    seguir = 's'
    
    while seguir == 's':
        movimientos = 0
        matrix = setup()
        while not solved(matrix):
            movimientos += 1
            printMatrix(matrix)
            move(input('move (↑w, ←a, ↓s, →d): ').lower(), matrix)
            
        else:
            printMatrix(matrix)
            print('¡Felicidades!, has ganado')
            print('Movimientos: ', movimientos)
            seguir = input('¿Desea jugar otra vez? (s/n): ').lower()
    return movimientos

game_deslizante()


