#CREATED: 15/02/2024
#AUTHOR: Alejandro Alonso Sánchez
#SEMESTER: 2024-Ene-Jun
#COURSE: Bases de datos relacionales.
#DESCRIPTION: Crear una base de datos de carreas unviersitarias
DROP DATABASE IF EXISTS carreras;
CREATE DATABASE IF NOT EXISTS carreras;
USE carreras;

CREATE TABLE IF NOT EXISTS carrera
(
    car_id INT NOT NULL AUTO_INCREMENT,
	car_nombre VARCHAR(70) NOT NULL,
    car_abrv VARCHAR(8) NOT NULL COMMENT 'Iniciales',
    car_division VARCHAR(7) NOT NULL COMMENT 'Iniciales',
    car_sede VARCHAR(30) NOT NULL,
    PRIMARY KEY (car_id),
    INDEX idx_car_nombre(car_nombre),
    INDEX idx_car_division(car_division),
    UNIQUE uni_car_nombre_divisio(car_nombre, car_division)
);

CREATE TABLE IF NOT EXISTS estudiante
(
	est_id INT NOT NULL AUTO_INCREMENT,
    est_nombre VARCHAR(50) NOT NULL,
    est_ap_pat VARCHAR(35) NOT NULL,
    est_ap_mat VARCHAR(35),
    est_correo VARCHAR(50) NOT NULL,
    est_semestre TINYINT UNSIGNED NOT NULL,
	est_car_id INT,
    PRIMARY KEY (est_id),
    CONSTRAINT fk_est_cit
		FOREIGN KEY (est_car_id)
        REFERENCES carrera (car_id),
	INDEX idx_est_nombre (est_ap_pat, est_ap_mat, est_nombre),
    UNIQUE uni_est_correo (est_correo)
);

# Práctica de alter table:

# 1. Cambia el nombre del atributo car_abrv por car_ini en la tabla carrera.
#ALTER TABLE carrera RENAME COLUMN car_abrv TO car_ini;

# 2. Cambia el tipo de dato del atributo car_nombre a VARCHAR(50) NOT NULL en la tabla carrera.
#ALTER TABLE carrera MODIFY car_nombre VARCHAR(50) NOT NULL;

# 3. Agrega la columna est_telefono después de la columna est_correo en la tabla estudiante, define tú el tipo y hazlo NOT NULL.
#ALTER TABLE estudiante ADD  est_telefono CHAR(10) NOT NULL;

# 4. Agrega el índice UNIQUE (condición de unicidad) para est_telefono en la tabla estudiante.
#ALTER TABLE estudiante ADD UNIQUE uni_telefono(est_telefono);

# 5. Elimina el índice división en la tabla carrera.
#ALTER TABLE carrera DROP INDEX idx_car_division;

-- Inserción de carreras

-- Inserción de carreras para DICIS
INSERT INTO carrera (car_nombre, car_abrv, car_division, car_sede) 
VALUES ('Artes Digitales', 'LAD', 'DICIS', 'Campus irapuato salamanca'),
       ('Ingeniería de datos e inteligencia artificial', 'LIDIA', 'DICIS', 'Campus irapuato salamanca');

-- Inserción de carreras para DCEA
INSERT INTO carrera (car_nombre, car_abrv, car_division, car_sede) 
VALUES ('Ingeniería', 'Ing', 'DCEA', 'Campus Principal'),
       ('Ciencias de la Computación', 'CiComp', 'DCEA', 'Campus Principal');

-- Inserción de carreras para DICIVA
INSERT INTO carrera (car_nombre, car_abrv, car_division, car_sede) 
VALUES ('Artes Visuales', 'ArtVi', 'DICIVA', 'Campus de Arte'),
       ('Producción Musical', 'ProdMus', 'DICIVA', 'Campus de Arte');

-- Inserción de estudiantes para las carreras
-- Para carreras de DCEA
INSERT INTO estudiante (est_nombre, est_ap_pat, est_ap_mat, est_correo, est_semestre, est_car_id) 
VALUES ('Juan', 'Pérez', 'Sánchez', 'juan.perez@ugto.mx', 3, 1), -- Ingeniería
       ('Ana', 'García', 'Martínez', 'ana.garcia@ugto.mx', 2, 1), -- Ingeniería
       ('María', 'López', '', 'maria.lopez@ugto.mx', 4, 2); -- Ciencias de la Computación

-- Para carreras de DICIS
INSERT INTO estudiante (est_nombre, est_ap_pat, est_ap_mat, est_correo, est_semestre, est_car_id) 
VALUES ('Miguel', 'Hernández', '', 'miguel.hernandez@ugto.mx', 5, 3), -- Artes Digitales
       ('Elena', 'Rodríguez', '', 'elena.rodriguez@ugto.mx', 3, 3), -- Artes Digitales
       ('Óscar', 'Gómez', '', 'oscar.gomez@ugto.mx', 4, 4); -- Ingeniería de datos e inteligencia artificial

-- Para carreras de DICIVA
INSERT INTO estudiante (est_nombre, est_ap_pat, est_ap_mat, est_correo, est_semestre, est_car_id) 
VALUES ('Sofía', 'Fernández', '', 'sofia.fernandez@ugto.mx', 2, 5), -- Artes Visuales
       ('Guillermo', 'Díaz', '', 'guillermo.diaz@ugto.mx', 6, 5), -- Artes Visuales
       ('Emilia', 'Martínez', '', 'emilia.martinez@eugto.mx', 3, 6); -- Producción Musical
       
SELECT * FROM carrera WHERE car_nombre NOT LIKE '%Ingeniería%' AND car_division != 'DICIS' AND car_division != 'DICIVA';
SELECT * FROM carrera WHERE car_sede = 'Campus irapuato salamanca'

