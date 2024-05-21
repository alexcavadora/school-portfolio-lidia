#CREATED: 15/02/2024
#AUTHOR: Alejandro Alonso Sánchez
#SEMESTER: 2024-Ene-Jun
#COURSE: Bases de datos relacionales.
#DESCRIPTION: INTRO DE FUNCIONES Y OPERADORES COMUNES

# % MODULO
# * Multiplicación
# + Suma
# - Resta
# - Cambio de signo
# / División


USE catalogo_peliculas;

#Multiplicación
SELECT ele_salario * 1000 AS total 
	FROM elenco;

#Resta 
SELECT ele_salario - 1000 AS total 
	FROM elenco;

#Resta unitaria
SELECT - ele_salario AS total 
	FROM elenco;

#Division
SELECT ele_salario/1.5 AS total 
	FROM elenco;

#Division entera
SELECT ele_salario DIV 1.5 AS total 
	FROM elenco;

#MODULO
SELECT ele_salario % 1.5 AS total, 
	ele_salario MOD 1.5 AS total2
	FROM elenco;

#FUNCIONES MATEMÁTICAS
#ROUND
SELECT ROUND(ele_salario /1.5) AS total, # NO DECIMAL
	ROUND(ele_salario /1.5, 2) AS total2 # 2 DECIMAL
	FROM elenco;

#SQUARE ROOT
SELECT SQRT(ele_salario) AS total, # Sqrt
	ROUND(SQRT(ele_salario), 2) AS total2 # Ronded sqrt
	FROM elenco;
    
#Cuando operamos o evaluamos funciones en valores no numéricos, se devuelve 0

SELECT ele_personaje * 2  from elenco;

SELECT SQRT(-ele_salario) AS total
	FROM elenco;
    
#cuando matemáticamente se encuentra un error, se devuelve -0 o null
#FUNCIONES DE AGREGACIÓN
# conteo de elementos
SELECT COUNT(*)
	FROM elenco;
    
# máximo de la selección
SELECT MAX(ele_salario)
	FROM elenco;

# mínimo de la selección
SELECT MIN(ele_salario)
	FROM elenco;

# máximo de la selección
SELECT MAX(ele_personaje)
	FROM elenco;

# sumatoria
SELECT SUM(ele_salario)
	from elenco;
    
# promedio
SELECT AVG(ele_salario)
	from elenco;

# desviación estándar
SELECT STD(ele_salario)
	from elenco;

# Funciones de strings

#CHAR_LENGTH
SELECT act_nombre,
		LENGTH(act_nombre) AS longitud1,
		CHAR_LENGTH(act_nombre) AS longitud2
    FROM actor;

#concat 
SELECT	concat(act_nombre, ' --- ', act_nombre_real) AS nombre
    FROM actor;
    
#concat with separator
SELECT	CONCAT_WS(' --- ',act_nombre, act_nombre_real, act_pais) AS concatWS
    FROM actor;
    
#format number with decimal places

SELECT FORMAT(ele_salario,2) AS total
	FROM elenco;

# LOWER AND UPPER
SELECT act_nombre as original,
		LOWER(act_nombre) AS minusculas,
		UPPER(act_nombre) AS mayusculas
FROM actor;	
# reverse
SELECT act_nombre as original,
		REVERSE(act_nombre) AS revertido
FROM actor;	

#replace (s, f, t) replace from string f's into t's
SELECT REPLACE(act_nombre, 'D', 'holaaa')
	FROM actor;	
    
#substring (s, p, n) regresa substring size n from position p 
#substring (s, p) regresa substring desde p hasta el final
#substring (s, 1, n) regresa los primerso n caracteres
SELECT act_nombre as original,
	substr(act_nombre, 5, 2) as sub,
    substr(act_nombre, 5) as sub2,
    substr(act_nombre, 1, 5) as sub3
	FROM actor;	
    
#Funciones de casteo, transformar de un tipo a otro

SELECT ele_salario as original,
	CAST(ele_salario AS CHAR(10)) AS casteado
    FROM elenco;
    
SELECT NOW();

    
    