#CREATED: 15/02/2024
#AUTHOR: Alejandro Alonso Sánchez
#SEMESTER: 2024-Ene-Jun
#COURSE: Bases de datos relacionales.
#DESCRIPTION: Crear una base de datos de un sistema de películas 
DROP DATABASE IF EXISTS catalogo_peliculas;
CREATE DATABASE IF NOT EXISTS catalogo_peliculas;
USE catalogo_peliculas;

CREATE TABLE IF NOT EXISTS pelicula
(
	pel_id INT NOT NULL AUTO_INCREMENT,
	pel_titulo VARCHAR(50) NOT NULL,
    pel_fecha_estreno DATE NOT NULL,
    pel_genero VARCHAR(30) NOT NULL,
    pel_clas ENUM('AA','A','B','B15','C','D') NOT NULL,
    pel_duracion INT NOT NULL COMMENT 'MINUTOS',
    pel_director VARCHAR(50) NOT NULL,
    PRIMARY KEY(pel_id),
    INDEX idx_pel_titulo(pel_titulo),
    INDEX idx_pel_clas(pel_clas),
    INDEX idx_pel_genero(pel_genero),
    INDEX idx_pel_director(pel_director),
    UNIQUE uni_pel_titulo_estreno (pel_titulo, pel_fecha_estreno)
);
# ENUM es un tipo de dato que permite ingresar un valor al atributo a partir
# de un conjunto de valores definidos. Si le intento ingresar un valor
# que no está en la lista de valores válidos, me va a marcar error.
# La duración de las películas es una cantidad física.
# Para las cantidades físicas o medidas, vamos a utilizar # tipos numéricos: INT (TINYINT), FLOAT

# Cuando se trabajen con medidas, cantidades físicas o monetarias # o propiedades físicas, es importante definir las unidades de medida
# que se van a utilizar en los atributos. Esto es para evitar
# confusiones con el usuario y otros desarrolladores.
# Peso (kg, 1b, oz), anchos (m, in, ..), distancia (km, mill, ...)
# Valores monetarios: pesos mexicanos, dólares, soles, euros, bitcoin, ...
# Si no definen las unidades pueden suceder:
# - Errores al capturar la información.
# - Errores aritméticos.
# para indicar las unidades que vamos a utilizar, dentro de la definición
# de la estructura de la tabla, vamos a emplear COMMENT

#Decimal asegura que el valor representado sea exacto

CREATE TABLE IF NOT EXISTS actor 
(
	act_id INT NOT NULL AUTO_INCREMENT,
    act_nombre VARCHAR(30) NOT NULL,
    act_fecha_nac DATE NOT NULL,
    act_nombre_real VARCHAR(50),
    act_genero ENUM('H','M','Otro'),
    act_pais VARCHAR(40),
    PRIMARY KEY(act_id),
    INDEX idx_act_nombre(act_nombre),
    INDEX idx_act_pais(act_pais),
    UNIQUE uni_act_nombre_nombre_real(act_nombre, act_nombre_real),
    UNIQUE uni_act_nombre_nacimiento(act_nombre, act_fecha_nac)
);

CREATE TABLE IF NOT EXISTS elenco
(
	ele_act_id INT NOT NULL,
    ele_pel_id INT NOT NULL,
    ele_personaje VARCHAR(50) NOT NULL,
    ele_categoria ENUM('Protagonico','Soporte','Extra'),
    ele_salario DECIMAL(15,2),
	PRIMARY KEY  (ele_act_id, ele_pel_id),
    INDEX idx_ele_personaje(ele_personaje),
    INDEX idx_ele_categoria(ele_categoria),
    CONSTRAINT fk_ele_act_id
		FOREIGN KEY (ele_act_id)
        REFERENCES actor(act_id)
		ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT fk_ele_pel_id
		FOREIGN KEY (ele_pel_id)
        REFERENCES pelicula(pel_id)
		ON UPDATE RESTRICT
        ON DELETE RESTRICT
);

# ON DELETE RESTRAINT/NO ACTION no permite hasta lidiar con los conflictos creados
# ON DELETE SET NULL establece los valores de la tabla hijo como NULL en los valores que estén referenciando un registro eliminado de la tabla autor.
# ON DELETE CASCADE permite eliminar el regristro de la tabla padre y a su vez eliminar todos los registros que hacen referencia a él en las tablas hijo
# DECIMAL es un tipo de dato particular de SQL para manejar cantidades monetarias.
# Su formato es DECIMAL(d, p)
# d es la cantidad total de dígitos, y p es la precisión decimal.
# DECIMAL (15, 2) -> 9999999999999.99 -> Valor máximo
# Total son 15 dígitos, 13 enteros y 2 decimales
# DECIMAL (15, 4) --> 99999999999.9999 --> Valor máximo 
# Toal son 15 dígitos, 11 enteros y 4 decimales


# 1. Crea un atributo llamado pel_presupuesto en la tabla película, define tú el tipo y hazlo NOT NULL.
#ALTER TABLE pelicula ADD pel_presupuesto DECIMAL(20,2) NOT NULL COMMENT 'MXN PESOS';

# 2. Elimina el índice de categoría en la tabla elenco.
#ALTER TABLE elenco DROP INDEX idx_ele_categoria;

# 3. Elimina la propiedad de unicidad de los atributos act_nombre, act_nombre_real en la tabla actor.
#ALTER TABLE actor DROP INDEX uni_act_nombre_nacimiento, DROP INDEX uni_act_nombre_nombre_real;

# 4. Crea un índice para el nombre real en la tabla actor.
#ALTER TABLE actor ADD INDEX idx_act_nom_real(act_nombre_real);

# 5. Cambia el nombre de la tabla elenco por reparto.
#ALTER TABLE elenco RENAME reparto;

# 6. Cambia el nombre y el tipo de dato del atributo pel_titulo por pel_nombre y tipo VARCHAR(40) NOT NULL.
#ALTER TABLE pelicula CHANGE pel_titulo pel_nombre VARCHAR(40);

-- Insertar películas
INSERT INTO pelicula (pel_titulo, pel_fecha_estreno, pel_genero, pel_clas, pel_duracion, pel_director) 
VALUES 
('El Padrino', '1972-03-24', 'Drama, Crimen', 'B15', 175, 'Francis Ford Coppola'),
('El Señor de los Anillos: El Retorno del Rey', '2003-12-17', 'Aventura, Drama, Fantasía', 'AA', 201, 'Peter Jackson'),
('Parásitos', '2019-05-30', 'Comedia, Drama, Thriller', 'B15', 132, 'Bong Joon-ho'),
('Alien, el octavo pasajero', '1979-05-25', 'Ciencia ficción, Terror', 'C', 117, 'Ridley Scott'),
('El Club de la Pelea', '1999-10-15', 'Drama', 'B15', 139, 'David Fincher'),
('El Resplandor', '1980-06-13', 'Drama, Terror', 'D', 144, 'Stanley Kubrick'),
('El Cisne Negro', '2010-09-01', 'Drama, Suspense', 'AA', 108, 'Darren Aronofsky'),
('Eterno resplandor de una mente sin recuerdos', '2004-03-19', 'Drama, Romance, Sci-fi', 'A', 108, 'Michel Gondry'),
('Pulp Fiction', '1994-09-10', 'Drama, Crimen', 'AA', 154, 'Quentin Tarantino'),
('La La Land', '2016-12-09', 'Comedia, Drama, Musical', 'A', 128, 'Damien Chazelle'),
('El Exorcista', '1973-12-26', 'Terror, Thriller', 'B15', 122, 'William Friedkin'),
('Forrest Gump', '1994-06-23', 'Comedia, Drama, Romance', 'A', 142, 'Robert Zemeckis');

-- Insertar actores
INSERT INTO actor (act_nombre, act_fecha_nac, act_nombre_real, act_genero, act_pais) 
VALUES 
('Marlon Brando', '1924-04-03', 'Marlon Brando Jr.', 'H', 'USA'),
('Al Pacino', '1940-04-25', 'Alfredo James Pacino', 'H', 'USA'),
('Robert De Niro', '1943-08-17', 'Robert Anthony De Niro Jr.', 'H', 'USA'),
('Scarlett Johansson', '1984-11-22', 'Scarlett Ingrid Johansson', 'M', 'USA'),
('Cate Blanchett', '1969-05-14', 'Catherine Elise Blanchett', 'M', 'Australia'),
('Tom Hanks', '1956-07-09', 'Thomas Jeffrey Hanks', 'H', 'USA'),
('Leonardo DiCaprio', '1974-11-11', 'Leonardo Wilhelm DiCaprio', 'H', 'USA'),
('Natalie Portman', '1981-06-09', 'Natalie Hershlag', 'M', 'Israel'),
('Emma Stone', '1988-11-06', 'Emily Jean Stone', 'M', 'USA'),
('Anthony Hopkins', '1937-12-31', 'Philip Anthony Hopkins', 'H', 'UK'),
('Kate Winslet', '1975-10-05', 'Kate Elizabeth Winslet', 'M', 'UK'),
('Daniel Radcliffe', '1989-07-23', 'Daniel Jacob Radcliffe', 'H', 'UK');

-- Insertar datos en la tabla elenco
INSERT INTO elenco (ele_act_id, ele_pel_id, ele_personaje, ele_categoria, ele_salario) 
VALUES 
(1, 1, 'Don Vito Corleone', 'Protagonico', 1000000.00),
(2, 1, 'Michael Corleone', 'Protagonico', 900000.00),
(3, 1, 'Vito Corleone joven', 'Soporte', 500000.00),
(4, 2, 'Aragorn', 'Protagonico', 1200000.00),
(5, 2, 'Frodo Bolsón', 'Protagonico', 1100000.00),
(6, 2, 'Samwise Gamgee', 'Soporte', 700000.00),
(7, 3, 'Gi Taek', 'Protagonico', 950000.00),
(8, 3, 'Park Dong-ik', 'Protagonico', 850000.00),
(9, 3, 'Lee Jeong-eun', 'Soporte', 600000.00),
(10, 4, 'Ellen Ripley', 'Protagonico', 1000000.00),
(11, 4, 'Ash', 'Soporte', 500000.00),
(12, 5, 'Tyler Durden', 'Protagonico', 1300000.00),
(1, 5, 'Narrador', 'Protagonico', 1200000.00),
(2, 5, 'Marla Singer', 'Soporte', 700000.00),
(3, 6, 'Jack Torrance', 'Protagonico', 1100000.00),
(4, 6, 'Wendy Torrance', 'Soporte', 600000.00),
(5, 6, 'Danny Torrance', 'Soporte', 600000.00),
(6, 7, 'Nina Sayers', 'Protagonico', 1000000.00),
(7, 7, 'Lily / Black Swan', 'Soporte', 600000.00),
(8, 8, 'Mia Dolan', 'Protagonico', 1300000.00),
(9, 8, 'Joel Barish', 'Protagonico', 1200000.00),
(10, 9, 'Vincent Vega', 'Protagonico', 1000000.00),
(11, 9, 'Jules Winnfield', 'Soporte', 500000.00),
(12, 10, 'Mia Dolan', 'Protagonico', 1300000.00),
(1, 10, 'Sebastian Wilder', 'Protagonico', 1200000.00),
(2, 11, 'Regan MacNeil', 'Protagonico', 1000000.00),
(3, 11, 'Chris MacNeil', 'Soporte', 600000.00),
(4, 12, 'Forrest Gump', 'Protagonico', 1300000.00),
(5, 12, 'Jenny Curran', 'Soporte', 700000.00),
(6, 1, 'Don Vito Corleone', 'Protagonico', 1000000.00),
(7, 2, 'Aragorn', 'Protagonico', 1200000.00),
(8, 4, 'Gi Taek', 'Protagonico', 950000.00),
(9, 4, 'Ellen Ripley', 'Protagonico', 1000000.00),
(10, 5, 'Tyler Durden', 'Protagonico', 1300000.00),
(11, 6, 'Jack Torrance', 'Protagonico', 1100000.00),
(12, 7, 'Nina Sayers', 'Protagonico', 1000000.00);

-- Borrar datos en la BD de películas:
-- Borrar registros en el elenco para película específica (dada por ti), que hayan tenido papel protagónico y cuyo salario sea mayor a 500mil.
DELETE 
	FROM elenco
    WHERE ele_salario > 500000.0 AND ele_pel_id = 4;
    
-- Borrar un actor cualquiera con id específico (dado por ti).
DELETE 
	FROM actor
    WHERE act_id =  1;


-- Borrar actores que hayan nacido entre 1985 y 1995 y sean de USA o de Reino Unido.
DELETE actor 
	FROM actor
    WHERE act_fecha_nac BETWEEN 1985-01-01 AND 1994-12-31;
    
SELECT * FROM actor;
-- Borrar películas cuyos títulos empiecen con las frase 'el hombre'.
-- Borrar las películas que duren menos de 110 minutos y sean del género 'acción'.

#********************************************
# *************** Subconsultas ***************
#********************************************
# Una subconsulta es una instrucción SELECT dentro de
# otra instrucción SELECT.

# Una subconsulta puede regresar un escalar (un único valor), # un registro (un único renglón), o una tabla (uno o más registros, # con una o más columnas).
# --› Tabla temporal
# La subconsulta es tomar el resultado del SELECT interno
# y utilizarlo (como tabla temporal) para otra consulta.
# Nota: El resultado de una subconsulta, también se
# puede utilizar en las instrucciones DELETE y UPDATE.
# Ejemplo 1: Devolver el título de las películas con la duración
# más larga de todas.

SELECT *
	FROM pelicula;

# Intento 1:
SELECT pel_titulo, pel_duracion
FROM pelicula
ORDER BY pel_duracion DESC
LIMIT 1;
# Ordena el resultado y devuelve la primera
# pero no devuelve todas las que tengan
# la duración máxima

# Intento 2:
SELECT pel_titulo, MAX(pel_duracion)
	FROM pelicula
	WHERE pel_duracion = MAX (pel_duracion);
# MAX no se puede usar en el WHERE
# Intento 3:
SELECT pel_titulo, MAX(pel_duracion)
	FROM pelicula;
    
# Se devuelve siempre el mínimo de registros # de acuerdo con la función o el criterio
# de búsqueda.
# La función MAX solo devuelve un valor (el valor
# máximo)
# Si lo pongo asi, MysQL en la columna pel_titulo
# me devuelve el valor de esa columna en el
# primer registro.
# MAX (pel_duracion) sí me devuelve la duración máxima

#intento 4
SELECT MAX(pel_duracion) AS dur_max
	FROM pelicula;

INSERT INTO pelicula (pel_titulo, pel_fecha_estreno, pel_genero, pel_clas, pel_duracion, pel_director) 
VALUES 
('Ejemplo 201', '1972-03-24', 'Drama, Crimen', 'B15', 201, 'Francis Ford Coppola');
#intento 5 
SELECT pel_titulo, pel_duracion
	FROM pelicula
	WHERE pel_duracion = (SELECT MAX(pel_duracion) AS dur_max FROM pelicula);

# Nota: Las funciones agregativas (MAX, MIN, AVG, SUM)
# devuelven siempre UN SOLO VALOR, ya sea en general
# O UNO PARA CADA GRUPO CUANDO SE UTILIZA GROUP BY

SELECT ele_act_id, ele_salario
	FROM elenco
	WHERE ele_salario = (SELECT MAX(ele_salario) AS sal_mas FROM elenco);
# Paso 5: Unir la tabla resultante con la tabla actor
SELECT act_id, act_nombre, elenco.ele_salario
	FROM actor
	INNER JOIN elenco
		ON act_id = elenco.ele_act_id
	WHERE elenco.ele_salario = (SELECT MAX(ele_salario) AS sal_mas FROM elenco);

# Ejercicio 3: Recupera el título de las películas que hayan
# pagado el salario más alto de todas.

SELECT pel_titulo, ele_salario
	FROM pelicula
    INNER JOIN elenco
		ON pel_id = elenco.ele_pel_id
	WHERE elenco.ele_salario = (SELECT MAX(ele_salario) FROM elenco);

SELECT DISTINCT pelicula.pel_titulo, elenco.ele_salario
	FROM pelicula
	INNER JOIN elenco
		ON pelicula.pel_id = elenco. ele_pel_id
	WHERE elenco. ele_salario = (SELECT MAX(ele_salario) AS sal_max
	FROM elenco);
# Con DISTINCT recupera cada combinación única
# para eliminar repetidos.
# Ejercicio 4: Recupera el nombre de los actores
# que participaron en las películas con el título
# más largo de todas (en caracteres).

# CHAR_LENGTH() 

SELECT pel_id, pel_titulo
	FROM pelicula
	INNER JOIN (SELECT MAX(CHAR_LENGTH(pel_titulo))AS mas_largo FROM pelicula) as t
		ON CHAR_LENGTH(pel_titulo) = t.mas_largo;
    
SELECT pel_titulo, CHAR_LENGTH(pel_titulo), actor.act_nombre
FROM pelicula
INNER JOIN (SELECT MAX(CHAR_LENGTH(pel_titulo)) AS mas_largo
FROM pelicula) AS t2
ON CHAR_LENGTH(pel_titulo) = t2.mas_largo
INNER JOIN elenco
ON pelicula.pel_id = elenco.ele_pel_id
INNER JOIN actor
ON actor.act_id = elenco.ele_act_id;



SELECT *
	FROM elenco;
   
#maximo salario en cada película
SELECT elenco.ele_pel_id, MAX(elenco.ele_salario) as sal_max
	FROM elenco
		GROUP BY ele_pel_id;
  
#id de actor con salario máximo de cada película
SELECT elenco.ele_pel_id, elenco.ele_act_id, elenco.ele_salario
	FROM elenco 
    INNER JOIN (SELECT elenco.ele_pel_id, MAX(elenco.ele_salario) AS sal_max
					FROM elenco
					GROUP BY ele_pel_id) AS t2
	ON t2.sal_max = elenco.ele_salario 
    AND t2.ele_pel_id = elenco.ele_pel_id;

#agregar al actor para obtener el nombre en la tabla

    SELECT elenco.ele_pel_id, elenco.ele_act_id, elenco.ele_salario, actor.act_id, elenco.ele_personaje
	FROM elenco 
    INNER JOIN (SELECT elenco.ele_pel_id, MAX(elenco.ele_salario) AS sal_max
					FROM elenco
					GROUP BY ele_pel_id) AS t2
	ON t2.sal_max = elenco.ele_salario 
    AND t2.ele_pel_id = elenco.ele_pel_id
    INNER JOIN actor
		ON 	elenco.ele_act_id = actor.act_id;