#CREATED: 15/02/2024
#AUTHOR: Alejandro Alonso Sánchez
#SEMESTER: 2024-Ene-Jun
#COURSE: Bases de datos relacionales.
#DESCRIPTION: Crear una base de datos de una empresa por departamento
DROP DATABASE IF EXISTS gestion_empleados;
CREATE DATABASE IF NOT EXISTS gestion_empleados;
USE gestion_empleados;
CREATE TABLE IF NOT EXISTS departamento
(
	dep_id INT AUTO_INCREMENT NOT NULL,
    dep_nombre VARCHAR(60) NOT NULL,
    dep_description TINYTEXT, # Permite ingresar Strings de longitud mayor a la permitida por varchar.
    dep_ubicacion ENUM('A','B','C','D'),
    PRIMARY KEY (dep_id),
    UNIQUE uni_dep_nombre(dep_nombre),
    INDEX idx_dep_ubicacion(dep_ubicacion),
    UNIQUE uni_dep(dep_ubicacion,dep_nombre)
);

CREATE TABLE IF NOT EXISTS empleado
(
	emp_id INT AUTO_INCREMENT NOT NULL,
    emp_nombre VARCHAR(30) NOT NULL,
    emp_ap_pat VARCHAR(30) NOT NULL,
    emp_ap_mat VARCHAR(30),
    emp_direccion VARCHAR(50) NOT NULL,
    emp_telefono CHAR(10) NOT NULL, # Almacena una cantidad fija de caracteres, si no se almacena la cantidad esperada, se llena con espacios.
    emp_correo VARCHAR(50) NOT NULL,
    PRIMARY KEY (emp_id),
    INDEX idx_emp_nom(emp_ap_pat, emp_ap_mat, emp_nombre),
    INDEX idx_emp_dir(emp_direccion),
    UNIQUE uni_correo(emp_correo),
    UNIQUE uni_telefono(emp_telefono),
    UNIQUE uni_emp_nom_dir(emp_ap_pat, emp_ap_mat, emp_nombre, emp_direccion)
);

CREATE TABLE IF NOT EXISTS asignacion
(
	asi_id INT AUTO_INCREMENT NOT NULL,
    asi_fecha_ini DATE NOT NULL,
    asi_fecha_fin DATE COMMENT 'DEJE VACIO PARA FECHA INDEFINIDA',
    asi_puesto VARCHAR(40) NOT NULL,
    asi_salario DECIMAL (15,2) NOT NULL COMMENT 'MXN PESOS',
    asi_emp_id INT NOT NULL,
    asi_dep_id INT,
    PRIMARY KEY (asi_id),
    INDEX idx_asi_fecha(asi_fecha_ini),
    INDEX idx_asi_puesto(asi_puesto),
    UNIQUE uni_asi_emp_fecha(asi_emp_id, asi_puesto, asi_fecha_ini),
	CONSTRAINT fk_asi_emp_id
		FOREIGN KEY (asi_emp_id)
        REFERENCES empleado(emp_id)
		ON DELETE CASCADE
        ON UPDATE CASCADE,
	CONSTRAINT fk_asi_dep_id
		FOREIGN KEY (asi_dep_id)
        REFERENCES departamento(dep_id)
        ON DELETE RESTRICT
        ON UPDATE RESTRICT #No puedo borrar un departamento si hay asignaciones pendientes
    
);

# 1. De la tabla asignación elimina el índice del atributo asi_puesto.
#ALTER TABLE asignacion DROP asi_puesto;

# 2. De la tabla departamento cambia el tipo de dato del atributo dep_descripcion a un varchar(100).
#ALTER TABLE departamento MODIFY dep_description VARCHAR(100);

# 3. De la tabla empleado elimina el atributo emp_correo.
#ALTER TABLE empleado DROP emp_correo;

# 4.Para la tabla asignación agrega un índice sobre el atributo asi_fecha_fin,
#ALTER TABLE asignacion ADD INDEX idx_fecha_fin(asi_fecha_fin);

#5.- En la tabla asignación agrega el atributo asi_total_horasz con un tipo de dato entero, y un comentario 
#en el que se especifique que el atributo hace referencia al total de horas invertido en la asignación. 
#El atributo agregado debe estar entre el atributo asi_fecha_fin y el atributo asi_puesto. 

#ALTER TABLE asignacion 
#	ADD asi_puesto VARCHAR(40) NOT NULL AFTER asi_fecha_fin, 
#    ADD asi_total_horasz INT COMMENT 'Hr invertidas en la asignación' AFTER asi_fecha_fin


-- Insertar 2 departamentos para cada ubicación
INSERT INTO departamento (dep_nombre, dep_description, dep_ubicacion)
VALUES 
('Departamento 1A', 'Descripción del Departamento 1A', 'A'),
('Departamento 2A', 'Descripción del Departamento 2A', 'A'),
('Departamento 1B', 'Descripción del Departamento 1B', 'B'),
('Departamento 2B', 'Descripción del Departamento 2B', 'B'),
('Departamento 1C', 'Descripción del Departamento 1C', 'C'),
('Departamento 2C', 'Descripción del Departamento 2C', 'C'),
('Departamento 1D', 'Descripción del Departamento 1D', 'D'),
('Departamento 2D', 'Descripción del Departamento 2D', 'D');

-- Insertar 7 empleados
INSERT INTO empleado (emp_nombre, emp_ap_pat, emp_ap_mat, emp_direccion, emp_telefono, emp_correo)
VALUES 
('Empleado1', 'Apellido1', 'Apellido2', 'Dirección1', '1234567890', 'empleado1@example.com'),
('Empleado2', 'Apellido1', 'Apellido2', 'Dirección2', '1234567891', 'empleado2@example.com'),
('Empleado3', 'Apellido1', 'Apellido2', 'Dirección3', '1234567892', 'empleado3@example.com'),
('Empleado4', 'Apellido1', 'Apellido2', 'Dirección4', '1234567893', 'empleado4@example.com'),
('Empleado5', 'Apellido1', 'Apellido2', 'Dirección5', '1234567894', 'empleado5@example.com'),
('Empleado6', 'Apellido1', 'Apellido2', 'Dirección6', '1234567895', 'empleado6@example.com'),
('Empleado7', 'Apellido1', 'Apellido2', 'Dirección7', '1234567896', 'empleado7@example.com');

-- Insertar 2 asignaciones para cada empleado
INSERT INTO asignacion (asi_fecha_ini, asi_fecha_fin, asi_puesto, asi_salario, asi_emp_id, asi_dep_id)
VALUES 
('2022-01-01', '2022-12-31', 'Puesto1', 25000.00, 1, 1),
('2023-01-01', NULL, 'Puesto2', 30000.00, 1, 2),
('2022-02-01', '2023-01-31', 'Puesto1', 28000.00, 2, 3),
('2023-02-01', NULL, 'Puesto2', 32000.00, 2, 4),
('2022-03-01', '2022-12-31', 'Puesto1', 26000.00, 3, 5),
('2023-03-01', NULL, 'Puesto2', 31000.00, 3, 6),
('2022-04-01', '2023-03-31', 'Puesto1', 27000.00, 4, 7),
('2023-04-01', NULL, 'Puesto2', 33000.00, 4, 8),
('2022-05-01', '2022-12-31', 'Puesto1', 25500.00, 5, 1),
('2023-05-01', NULL, 'Puesto2', 30500.00, 5, 2),
('2022-06-01', '2023-05-31', 'Puesto1', 27500.00, 6, 3),
('2023-06-01', NULL, 'Puesto2', 31500.00, 6, 4),
('2022-07-01', '2022-12-31', 'Puesto1', 26500.00, 7, 5),
('2023-07-01', NULL, 'Puesto2', 32500.00, 7, 6);



