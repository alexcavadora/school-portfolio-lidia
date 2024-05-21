#CREATED: 12/02/2024
#AUTHOR: Alejandro Alonso Sánchez
#SEMESTER: 2024-Ene-Jun
#COURSE: Bases de datos relacionales.
#DESCRIPTION: Crear una base de datos relacional para el manejo de la información de una agenda básica.

#Empezamos con DDL Data Definition Language
#Haremos un diagrama relacional a instrucciones de SQL, utilizando DDL
#SQL = DDL (Data definition language) + DML (Data manipulation language)
#SQL es un lenguaje interpretado, escribimos scripts pues no tiene un compilador.
#No genera un ejecutable sino se interpreta y ejecuta cada línea de código de una por una.
#Con el workbench nos conectamos a el kernel de MySQL
# (el intérprete) que a su vez está conectado al servidor de la base de datos.
#creando el edificio donde pondremos nuestra biblioteca
DROP DATABASE IF EXISTS agenda;
CREATE DATABASE IF NOT EXISTS agenda; # nombre de la biblioteca, la crea si no existe, de lo contrario, continuamos
USE agenda; # se habilita el uso de la base de datos para manipularlos
# USE sys; al cambiar de bases de datos, el menú a la izquierda se actualiza y la base abierta aparecerá en negritas.
# <------------
#Crear la tabla de contacto

CREATE TABLE IF NOT EXISTS contacto
(
	con_id INT AUTO_INCREMENT NOT NULL,
    con_nombre VARCHAR(50) NOT NULL,
    con_telefono VARCHAR(10) NOT NULL,
    con_direccion VARCHAR(60),
    con_correo VARCHAR(25) NOT NULL,
    PRIMARY KEY (con_id), #Esto indica los atributos que forman parte de la llave primario
						# Cuando la llave es un atributo tipo INT y AUTO_INCREMENT
						# Se le conoce como llave artificial
	INDEX idx_nombre(con_nombre), #Creación del índice del nombre
    UNIQUE uni_telefono (con_telefono), #Creación del unique uni_telefono
    UNIQUE uni_correo (con_correo)
);
#INT entero en un rango
#VARCHAR(50) String vari
#NOT NULL no acepta valores vacíos o nulos.
#AUTO_INCREMENT auto incremento de valores
#Toda tabla necesita una llave primaria, 
# Es usado especialmente para llaves primarias artificiales.
# No puede haber dos atributos auto incrementales en la misma tabla.
# EL primer valor de 1 cada que se inserta un nuevo registro.
# El valor nunca decrementa.


CREATE TABLE IF NOT EXISTS cita
(
	cit_id INT NOT NULL AUTO_INCREMENT,
	cit_lugar VARCHAR(60),
    cit_fecha DATE NOT NULL,
    cit_hora TIME NOT NULL,
    cit_con_id INT,
    PRIMARY KEY (cit_id),
    INDEX idx_fecha(cit_fecha),
    INDEX idx_con_id(cit_con_id),
    UNIQUE uni_fecha_hora(cit_fecha, cit_hora), #UNIQUE con 2 atributos
    CONSTRAINT fk_con_cit
		FOREIGN KEY (cit_con_id)
        REFERENCES contacto (con_id)
);


# DATE: es un tipo de dato particular de SQL para el manejo de fechas
# FORMAT: YYYY-MM-DD
# TIME: es un tipo de dato particular de SQL para el manejo de horas/tiempos.
# FORMAT: HHH:MM:SS Tiene 3 H para medir tiempos, ejemplo se diferencían por 150 horas
# La ventaja es kanejarlos como un todo y hacer operaciones con ellos.
# Nombre de tablas:
# camelCase: nombre real -> nombreReal
# usaremos el:
# snake_case: nombre real -> nombre_real
# snake_case: 
# INDEX ayuda a agilizar búsquedas considerando uno o varios atributos. alter
# se suelen indexar los atributos que se pueden utilizar de manera frecuente en búsquedas/consultas

# Datos insertados
# con_nombre  			con_correo 			no.registro
#------------------------------------------------------------
# Pedro 				ptr@ugto.mx			1
# Ana 					anrt@ugto.mx		2
# Pedro 				pdu@ugto.mx			3
# Beatriz 				blop@gmail.mx		4

# INDEX idx nombre(con_nombre)
# con_nombre		no.registro
# Ana 			-->	2
# Beatriz 		--> 4
# Pedro 		--> [1, 3]

# El índice ordena el atributo de alguna manera (ascendiente)
# y guarda el apuntador al registro para acceder a la información

# INDEX idx_nombre_correo(con_nombre, con_correo)
# con_nombre  			con_correo 			no.registro
#------------------------------------------------------------
# Ana 					anrt@ugto.mx		2
# Beatriz 				blop@gmail.mx		4
# Pedro 				pdu@ugto.mx			3
# Pedro 				ptr@ugto.mx			1

# Si bien es práctico indexar, también requiere espacio en memoria,
# Hay que decidir cuándo conviene indexar para aumentar el desempeño de búsquedas
# Sin utilizar tanto espacio de almacenamiento (SobreIndexar)

#Nota 2: Por definición, la llave primaria es un índice y es único:
#PRIMARY KEY (con_id) --> INDEX(con_id) + UNIQUE(con_id) + NOT_NULL

#UNIQUE es similar a la llave primaria pero no se considera como tal por:
#	1.Solo puede haber una llave primaria en cada tabla.
#	2. UNIQUE acepta valores null, y PRIMARY KEY no.

# Datos insertados
# con_nombre  			con_correo 			no.registro
#------------------------------------------------------------
# Pedro 				ptr@ugto.mx			1
# Ana 					anrt@ugto.mx		2
# Pedro 				pdu@ugto.mx			3
# Beatriz 				blop@gmail.mx		4

# UNIQUE (con_correo) similar a INDEX, los ordena por correo, sin embargo no acepta repetidos.
# con_nombre  			con_correo 			no.registro
#------------------------------------------------------------
# Pedro 				ptr@ugto.mx			1
# Ana 					anrt@ugto.mx		2
# Pedro 				pdu@ugto.mx			3
# Beatriz 				blop@gmail.mx		4

#si intento insertar
# con_nombre  			con_correo 			
#--------------------------------------------
# Patricia 				ptr@ugto.mx	

#marcaría error, ya que ptr@ugto.mx ya existe	

#Ahora, UNIQUE uni_nomcor (con_nombre, con_correo)
	
#si intento insertar
# con_nombre  			con_correo 			
#--------------------------------------------
# Patricia 				ptr@ugto.mx	  		--> Si permite ya que la combinación de nombre y correo no existe.
# Pedro 					ptr@ugto.mx			--> Marca error porque la combinación ya existe.


#UNIQUE es una restricción que para poder facilitar su funcionamiento
#Genera un index

### Llave foránea
# La llave foránea es una restricción de referencia, para mantener la integridad referencial.
# Esto significa que para poder insertar un valor que es la lalve foránea, ese valor debe existir
# en la tabla referenciada.

# Tabla que contiene la llave foránea: tabla hijo.
# Tabla que contiene la llave referenciada: table padre

# Para poder insertar un valor en la tabla hijo, se debe tener un valor a la tabla padre.
# Por definición, puede aceptar valores nulos.
#ALTER TABLE amigo RENAME contacto;

# En nuestra definición, si insertamos una cita sin con_id, decimos que esa cita no está asociada a ningún contacto.
#ALTER TABLE contacto DROP COLUMN con_direccion; # Elimina la columna con_direccion
#ALTER TABLE contacto DROP COLUMN con_telefono, DROP COLUMN con_correo;
# Hace dos operaciones de ALTER TABLE, separadas por ','
# Borra la columna con_telefono y luego la columna con_correo
#ALTER TABLE contacto ADD con_correo VARCHAR(30);
# Agregar la columna con_correo al final de la tabla
#ALTER TABLE contacto ADD con_telefono CHAR(10) NOT NULL AFTER con_nombre;
# Agrega la columna con_telefono después de la columna con_nombre
#ALTER TABLE contacto ADD con_direccion VARCHAR(60) FIRST;
# Agrega la columna con_direccion al inicio de la tabla
# Cuando elimino una columna de la cual depende un índice, # el índice también se elimina.

#ALTER TABLE contacto ADD INDEX idx_correo (con_correo); # Agrega un índice para con_correo
#ALTER TABLE contacto DROP INDEX idx_correo; # Elimina el índice idx_correo
# Cuando elimino un índice, las columnas asociadas a este
# no se eliminan.

#ALTER TABLE contacto ADD UNIQUE uni_telefono (con_telefono);
# Agrega el índice UNIQUE (condición de unicidad) para con_telefono
#ALTER TABLE contacto ADD UNIQUE uni_correo (con_correo);
# Agrega el índice UNIQUE (condición de unicidad) para con_correo

#ALTER TABLE contacto CHANGE con_correo_electronico con_correo VARCHAR(50) NOT NULL;
# Renombra la columna con_cor_elect a con_correo y le cambia el tipo de dato
#ALTER TABLE contacto MODIFY con_correo VARCHAR(30) NOT NULL;
# Cambia el tipo de dato para con_correo
#ALTER TABLE cita DROP PRIMARY KEY; # Elimina la llave primaria # No elimina el atributo, elimina la definición de la llave primaria.
# Antes de poder borrar la definición de la llave primaria, hay que asegugrarse
# que el atributo asociado a la PK no tenga la propiedad AUTO_INCREMENT,
# y además que no sirva de referencia para una llave foránea en otra tabla.
# En este caso tenemos que cambiar el tipo de dato del atributo
# para poder quitar la definición de la PK
#ALTER TABLE cita MODIFY cit_id VARCHAR(2);

#ALTER TABLE cita DROP PRIMARY KEY;
# Ahora ya podemos quitar la definición de la PK
# Regresando la llave primaria
#ALTER TABLE cita ADD PRIMARY KEY (cit_id), MODIFY cit_id INT NOT NULL AUTO_INCREMENT;
# Se cambia de nuevo el tipo de dato y las propiedades
# de la columna cit_id que será la llave primaria

#DROP TABLE contacto;
# elimina la tabla contacto, no permite ahora porque existe una llave foránea

#ALTER TABLE cita DROP FOREIGN KEY fk_con_cit;
#DROP TABLE IF EXISTS contacto, cita;

#tengamos mucho cuidado no borrar todo

##############################*####*####*##*####*#**##*#**#*##*#**#*############
# Hasta aquí hemos trabajado con el DDL (Data Definition Language). Lo usamos
# para crear y gestionar (manipular) la estructura de la BD.
##＃#######＃＃######＃#＃##＃##＃####＃#＃#######＃####＃##＃＃#＃#＃＃#＃＃###＃＃#＃#＃##＃#＃＃#＃###
#USE agenda;

# Ahora vamos a empezar a trabajar con el DML （Data Manipulation Language）. Lo
# usamos para crear y manipular (geftionar) los DATOS dentro de una estructura
# que ya definimos.
# La estructura de la BD se crea (casi siempre) solo 1 vez
# y se modifica algunas (pocas) veces.
# La manipulación de datos, ocurre todo el tiempo.

# En DML hay 4 operaciones que podemos hacer, CRUD (create, read, update and delete)

# Create -> INSERT
# Read -> SELECT
# Update -> UPDATE
# Delete -> DELETE

# INSERT
# Inserta (agregar, crear) datos en una table de una estructura definida (base de datos)
# - Debemos insertar primero los datos en las tablas independientes (aquellas que no
# 	tienen llaves foráneas)
# Ya que las tablas dependientes necesitan información de las tablas independientes. 
# Si esa información no existe, marcará error.

# Ya que contacto no tiene llave foránea, insertaremos ahí antes de agendar una cita.

INSERT INTO contacto (con_nombre, con_telefono, con_direccion, con_correo)
VALUES ('Raúl Uriel','4771234567','Yeso 224, Col. Estrella', 'ru.s@ugto.mx');
#con_nombre VARCHAR(50) NOT NULL, con_telefono VARCHAR(10) NOT NULL, con_direccion VARCHAR(60), con_correo VARCHAR(25) NOT NULL,
# No se inserta un valor para con_id porque es un atributo AUTO_INCREMENT
# Para este atributo, SQL genera un valor de manera automática cuando
# se inserta el registro.

# Al hacer la inserción se debe tomar en cuenta el tipo de dato de cada atributo.
# Para saber qué valores puedo darle.
INSERT INTO contacto (con_nombre, con_telefono, con_direccion, con_correo)
VALUES ('María del Rosario', '4449876543', 'Antares 453, Col. Estrella', 'mrh@ugto.mx');
#Probamos con 13 caracteres en lugar de 10contacto
INSERT INTO contacto (con_nombre, con_telefono, con_direccion, con_correo)
VALUES	('Zair Pedro', 				'4648759635', 'Zapato 23, Col. Centro', 	'zp@ugto.mx'),
		('Emilio Martínez',	 		'4795876935', 'Brisas 123 Col. Mares', 	'emy@gmail.com'),
        ('Helio Rafael González', 	'4641457896', 'Hidalgo 34, Col. Centro', 	'hr@gmail.com'),	
		('Paola Arreguín Meléndez', '4641257846', 'Cazadora 56, Col. Bella Vista', 'pam@ugto.mx'),
		('Pedro Hernández', 	   	'4778541234', 'Estrella 78, Col. Misiones', 	'phk@gmail.fom'),
		('Claudia Mendoza Huerta', 	'4425879635', 'Independencia 45, Col. Cimatario', 'cmh@ugto.mx');

# Los valores que se insertan deben estar en orden (correspondencia) con respecto a las
# columnas indicadas.
# Tipos de datos: En principio no se pueden insertar valores de un tipo en una columna de
# otro tipo (p.ej. INT --> VARCHAR). --> Pero, MYSQL intenta hacer la conversión entre tipos # de manera automática. A veces se puede, a veces no. Es mejor asegurarse del tipo de dato
# de cada columna e insertar valores válidos.
# No se puede insertar un valor en una columna AUTO_INCREMENT -> Marca error
#INSERT INTO contacto (con_id, con_nombre, con_telefono, con_direccion, con_correo)
#VALUES (9, 'Prueba', 'Prueba', '1234567890', 'Prueba', 'Prueba');
# Si no se indica el nombre de una columna para insertar datos.,
# cuando se inserte el registro, se le pondrá el valor por default
# a esa columna (generalmente, el valor NULL).
INSERT INTO contacto ( con_nombre, con_telefono, con_correo)
VALUES ( 'Prueba', '1234567890', 'Prueba');

SELECT * FROM contacto;

#Actualizar datos en la tabla 
UPDATE contacto
	SET con_telefono = '2225874569'
	WHERE con_id = 3; # Actualiza el teléfono para el contacto con_id =2
SELECT * FROM contacto;

UPDATE contacto
	SET con_telefono = '4421234567'
	WHERE con_nombre = 'Juan Escutia';
SELECT * FROM contacto;
# Actualiza el teléfono para el contacto con_nombre = 'Juan Escutia'
# Utilizamos el con_id para restringir la modificación

# --- Actualizar datos de la tabla cita
	UPDATE cita
	SET cit_fecha = ADDDATE(cit_fecha, 2);
# Agrega 2 días al atributo cit_fecha EN TODOS LOS REGISTROS (no hay una sección WHERE)
# ADDDATE (DATE/DATETIME, num_dias) agrega una cantidad num_dias de días a
# un atributo DATE/DATETIME.

# Los tipos DATE/DATETIME automáticamente consideran las fechas
# válidas de los canlendarios.
SELECT * FROM cita;
UPDATE cita
	SET cit_fecha = ADDDATE(cit_fecha, 2)
	WHERE cit_fecha = '2024-06-17';
# Agrega 2 días al atributo ci_fecha cuando cit_fecha = '2024-06-17'
SELECT * FROM cita;
UPDATE cita
	SET cit_fecha = ADDDATE(cit_fecha, 2)
# WHERE cit_id >= 3 AND cit_id <= 6;
	WHERE cit_id BETWEEN 3 AND 6;
# Agrega 2 días al atributo cit_fecha cuando cit_id está entre 3 y 6
# Sin la sección WHERE se actualizan todos los registros de la tabla
# Hay que tener cuidado con los atributos que se quieren actualizar,
# sobre todo cuando hay condiciones de unicidad.

#UPDATE contacto
#SET con_telefono = '1234567890' ;
# Intenta modificar el teléfono de todos los contactos, asignándoles
# el mismo valor (a todos).
# Error! Tenemos definido con_telefono como UNIQUE, no puede haber
# 2 contactos con el mismo teléfono.

# Lo mismo sucedería si intentamos actualizar una cita
# a una fecha y hora ya existentes.
# ********************* Operadores lógicos
# AND y
# OR o
# NOT no
UPDATE cita
SET cit_fecha = ADDDATE(cit_fecha, 1)
WHERE (cit_fecha = '2024-05-05') AND (cit_hora = '16:00:00');
# Agrega 1 día a cit_fecha para los registros que cumplan las 2 condiciones
UPDATE cita
SET cit_fecha = ADDDATE(cit_fecha, 1)
WHERE (cit_fecha = '2024-05-05') OR (cit_fecha = '2024-05-12');
# Agrega 1 día a cit_fecha para los registros que cumplan las 2 condiciones

UPDATE cita
SET cit_hora = ADDTIME(cit_hora, '-2:00:00')
WHERE (cit_fecha = '2024-05-06' AND cit_hora = '13:00:00')
OR (cit_fecha = '2024-06-19' AND cit_hora = '18:45:00');
# Quita 2 horas de las citas con las fechas y horas indicadas
# Hay que tener cuidado al actualizar en caso
# de la condiciones de unicidad.
# Esto también aplica para las llaves foráneas, en este
# caso debe hacerse referencia a un valor existente
UPDATE cita
	SET cit_con_id = 15
	WHERE (cit_con_id = 1)
		AND (cit_fecha = '2024-04-21')
		AND (cit_hora = '10:00:00');
# con_id = 15 no existe en la tabla contacto

UPDATE cita
	SET cit_fecha = ADDDATE(cit_fecha, 1)
	WHERE cit_fecha > '2024-06-01';
# Agrega 1 día a las cita con fechas posteriores a '2024-06-01'
UPDATE cita
	SET cit_fecha = ADDDATE(cit_fecha, 1)
	WHERE cit_fecha <= '2024-04-28' ;
# Agrega 1 día a las cita con fechas anteriores a '2024-06-29'
UPDATE cita
	SET cit_fecha = ADDDATE(cit_fecha, 1)
	WHERE cit_fecha != '2024-05-06';
# Agrega 1 día a las citas con fechas diferentes a '2024-05-06'

UPDATE cita
SET cit_fecka = ADDDATE(cit_fecha, 1)
WHERE NOT cit_fecha = '2024-05-06';
# Agrega 1 día a las citas con fechas que no sean iguales a '2024-05-06'
# ********************* Operadores de pertenencia
UPDATE cita
SET cit_fecha = ADDDATE(cit_fecha, 2)
WHERE cit_fecha BETWEEN '2024-04-02' AND '2024-04-23';
# Agrega 2 días a la fecha si la fecha se encuentra en el rango indicado
UPDATE cita
SET cit_fecha = ADDDATE(cit_fecha, 1)
WHERE cit_fecha IN ('2024-04-02', '2024-05-06', '2024-06-21');



# Hay dos comodines en el operador LIKE:
# El comodín '%' considera cualquier string (cualquier cantidad de caracteres)
# (incluido el caracter nulo)
# El comodín '_' considera cualquer caracter (solo 1 caracter)
# no incluye el caracter nulo
# El resultado del operador LIKE es un valor boolean (True/False)
# Ejemplos de patrones:
# '%Alta' --> El string debe terminar en la palabra 'Alta', antes de esa palabra
# '_Alta_'--> puede venir cualquier cantidad de caracteres (los que sean)
# '_Alta' --> El string debe terminar en la palabra 'Alta', antes de esa palabra puede venir solo 1 caracter (cualquiera)
# '%Alta%' --> Cualquier cosa, palabra 'Alta', cualquier cosa
# '_Alta_' --> Un caracter, palabra 'Alta', un caracter
# '__Alta__' --› Dos caracteres, palabra 'Alta', dos caracteres)

# Patrones (Cómo se evalúan con el operador LIKE)
# ---------------------------------------------------------------------------------------------------------------
# cit_lugar				 '%Alta'		 '_Alta'		'%Alta%'		'_Alta_'	 '__Alta__'		'___Alta_'
# ---------------------------------------------------------------------------------------------------------------
# 'Vía Alta'				T				F				T				F			F				F
# 'aAlta'					T				T				T				F			F 				F			
# 'Vía Alta Zaragoza'		F				F				T				F 			F				F
# 'bbAltabb'				F				F				T 				F 			T 				F

# Una vez borrados o actualizados los datos, no se pueden recueprar
# o volver a un estado previo.
# Formar de mitigar ese impacto:
#a. Preguntar al usuario si está seguro de las operaciones (en la construcción del sistema)
# b. Crear un respaldo de los datos en un momento determinado (programado)
#c. Guardar los datos en un caché para poder recuperarlos en caso de que se requiera.
###--- Borrar y actualizar datos de una tabla, usando datos de otra tabla
# Eliminar datos de la tabla cita, usando información de la tabla contacto
DELETE contacto, cita # Estas son las tablas que voy a usar en total
FROM cita # Esta es la tabla en donde quiero borrar
INNER JOIN contacto ON contacto.con_id = cita.cit_con_id
	WHERE contacto.con_nombre LIKE '%ez';


