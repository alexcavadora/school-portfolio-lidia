# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:12:08 2023
@date: 10 nov 2023
@subject: Data Analysis Fundamentals
@author: Alejandro Alonso Sáncez
@description:
    Escribe un programa para simular el juego del ‘ahorcado’.Considera los siguientes
    aspectos del juego: Crea de forma manual un diccionario, en donde las llaves sean
    palabras y los valores sean definiciones cortas de esas palabras. Las palabras
    deben estar en  minúsculas  y  se consideran  acentos. El  diccionario  debe
    contener  al menos 20 palabras .p.ej.{‘casa’: ‘Edificio para habitar’,‘músico’:
   ‘Persona que conoce el arte de la música o lo ejerce’, ...}
    El programa debe seleccionar una palabra al azar del diccionario y pedirle al
    usuario que la adivine. El juego debe iniciar con una serie de _ separados por
    espacios que reemplacen las letras de la palabra seleccionada .p.ej.
    Adivina la palabra que tengo: _ _ _ _ 
    En el ejemplo, la palabra(‘casa’)está compuesta por 4 letras. Se debe pedir una
    letra al usuario. Si ésta pertenece a la palabra, el 
    programa debe mostrar las posiciones en las que se encuentra p.ej.
    input: ‘a’output:_ a _ a
    Si la letra no forma partede la palabra, se considera como 1 error.
    Se repite el ciclo para preguntar más letras al usuario, hasta que adivine 
    la palabra o alcance el límite de errores. 
    El límite es 8 errores.
    Cada error corresponde a una parte del ‘ahorcado’. Si se comete un error, se
    puede ir desplegando esa figura (como se muestra a continuación). 
    También es válido solamente colocar un contador de errores.

    Las  partes  del ‘ahorcado’son: viga  superior, soga,  cabeza, torso,  brazo
    izquierdo, brazo derecho, pierna izquierda, pierna derecha.•Si el usuario 
    adivina la palabra antes de los 8 errores, desplegar ladefinición de la palabra
    y enviar un mensaje de felicitación. p.ej.
    casa: Edificio para habitar 
    ¡Felicidades! Has ganado.
    Si el usuario no adivina la palabra antes de los 8 errores, enviarun mensaje
    de consolación.
    En ambos casos se debe preguntarsi quiere jugar de nuevo(s/n). En caso de ‘s’
    repetir el ciclo del juego. En cas ode ‘n’enviar un mensaje de despedida. Las
    letras  que  ingresa  el  usuario  se  deben  transformar  a  minúsculas  y  
    se deben ignorar losacentos. Es decir, si el usuario ingresa una ‘u’, y esa 
    letra está acentuada en la palabra aadivinar(p.ej. ‘músico’), se debe considerar 
    correcta y desplegar la letra correspondiente.
"""
import random as rn

palabras = {
    'Reflexión': 'Acción y efecto de reflejar o reflejarse',
    'Alejandro': 'Nombre del autor de este código',
    'Naranja': 'Fruto del naranjo. Color que representa dicho fruto.',
    'Murciélago': 'Acción y efecto de reflejar o Quiróptero insectívoro que tiene fuertes caninos y los molares con puntas cónicas.',
    'Camello': 'Mamífero artiodáctilo rumiante, oriundo del Asia central, corpulento y más alto que el caballo, con el cuello largo, la cabeza proporcionalmente pequeña y dos gibas en el dorso, formadas por acumulación de tejido adiposo.',
    'Asociación': 'Conjunto de los asociados para un mismo fin y, en su caso, persona jurídica por ellos formada.',
    'Hamburguesa': 'Pieza de carne picada aplastada y con forma redondeada, mezclada con diversos ingredientes, que se hace a la plancha, a la parrilla o frita.',
    'Alemania': 'País europeo.',
    'Diccionario': 'Repertorio en forma de libro o en soporte electrónico en el que se recogen, según un orden determinado, las palabras o expresiones de una o más lenguas, o de una materia concreta, acompañadas de su definición, equivalencia o explicación.',
    'Lista': 'Enumeración, generalmente en forma de columna, de personas, cosas, cantidades, etc., que se hace con determinado propósito.',
    'Entero': 'Número que consta exclusivamente de una o más unidades positivas o negativas, sin parte decimal, a diferencia de los quebrados y de los mixtos.',
    'Aeropuerto': 'Área destinada al aterrizaje y despegue de aviones dotada de instalaciones para el control del tráfico aéreo y de servicios a los pasajeros.',
    'Especialización': 'Acción y efecto de especializar o especializarse.',
    'Código': 'Conjunto de normas legales sistemáticas que regulan unitariamente una materia determinada.',
    'Manuscrito': 'Texto escrito a mano, especialmente el que tiene algún valor o antigüedad, o es de mano de un escritor o personaje célebre.',
    'Publicación': 'Escrito impreso, como un libro, una revista, un periódico, etc., que ha sido publicado.',
    'Maya': 'Dicho de una persona: De un antiguo pueblo que habitó desde la mitad sur de México hasta Honduras y hoy principalmente en Guatemala, Yucatán y otras regiones adyacentes.',
    'Polo': 'Cada uno de los dos puntos en que el eje de rotación corta un cuerpo esférico, especialmente la Tierra.',
    'México': 'El mejor país del mundo, conocido por ser megadiverso, con una increíble cantidad de platillos, culturas y tradiciones traídas y preservadas desde épocas prehispánicas.',
    'Estudiante': 'Persona que estudia.',
    'Huracán': 'Viento muy impetuoso y temible que, a modo de torbellino, gira en grandes círculos, cuyo diámetro crece a medida que avanza apartándose de las zonas de calma tropicales, donde suele tener origen.',
    }
#inicializacion para ciclos while
seguir_jugando = 's'
caracter = '  '
moniyo=['\n\n\n\n',
        '   -----\n\n\n\n',
        '   -----\n      |\n\n\n',
        '   -----\n      |\n      O\n\n',
        '   -----\n      |\n      O\n      |\n',
        '   -----\n      |\n      O\n     /|\n',
        '   -----\n      |\n      O\n     /|\\\n',
        '   -----\n      |\n      O\n     /|\\\n     /',
        '   -----\n      |\n      O\n     /|\\\n     / \\']

while seguir_jugando == 's':
    #escoger palabra al azar
    palabra, definicion = rn.choice(list(palabras.items()))
    
    #se almacena una version de la palabra en minúsculas, y sin acentos para facilitar el juego
    palabra_limpia = palabra.lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')
    
    
    #reiniciar las letras ingresadas para un nuevo juego
    letras_ingresadas = ''
    
        
    #se reinician los intentos en el nuevo juego
    intentos = 0
    
    #Se muestra la primera vez la palabra a adivinar
    print('Adivina la siguiente palabra, usando un caracter a la vez:')
    for i in palabra:
        print('_', end=' ') #end, nos permte cambiar el final que usualmente es una nueva linea por un espacio
    print('\n')
    
    #condicion de derrota
    while intentos != 8:
        #Leer el siguiente caracter, debe ser longitud 1 para que no haga trampas
        while(len(caracter) != 1):
            caracter = input('Introduce la siguiente letra: ').lower()
       
        #detectar si le atinó a la letra o no
        if caracter not in palabra_limpia:
            intentos += 1
        else:
           letras_ingresadas += caracter
           
        caracter = '  '
        
        #imprimir las letras adivinadas o en su defecto un guión
        for i in range(len(palabra_limpia)):
            if palabra_limpia[i] in letras_ingresadas:
                print(palabra[i], end = ' ')
            else:
                print('_', end = ' ')
        
        #imprime un monito en base a un arreglo de sus 'imágenes'
        print('\n\n', moniyo[intentos], '\n')
        
        #en caso de lograr ganar se muestra victoria
        if set(letras_ingresadas) == set(palabra_limpia):
            print(f'¡Felicitaciones!, has ganado. La palabra era: {palabra}. \nSe puede definir como: {definicion}')
            break
    else:
        #mensaje de derrota
        print(f'Lo siento, te has quedado sin intentos, la palabra era {palabra}.')
    seguir_jugando = ''
    
    while seguir_jugando != 's' and seguir_jugando != 'n':
        seguir_jugando = input('\n¿Desea seguir jugando? (s/n): ').lower()
    