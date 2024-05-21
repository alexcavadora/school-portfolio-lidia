# CREATED: 29/02/2024
# CREATED BY: Alejandro Alonso Sánchez
# Semester: 2024-EJ
# Course: Bases de Datos Relacionales
# Description: crear una bd que controle el sistema de citas de una clinica
DROP DATABASE IF EXISTS clinica;
CREATE DATABASE IF NOT EXISTS clinica;
USE clinica;

CREATE TABLE IF NOT EXISTS paciente
(
	pac_id INT NOT NULL AUTO_INCREMENT, 
    pac_nombre VARCHAR(25) NOT NULL,
    pac_ap_pat VARCHAR(25) NOT NULL,
    pac_ap_mat VARCHAR(25),
    pac_correo VARCHAR(25) NOT NULL,
    pac_telefono CHAR(10) NOT NULL,
    PRIMARY KEY (pac_id),
    INDEX idx_nombcomp(pac_ap_pat,pac_ap_mat,pac_nombre),
    UNIQUE uni_datospac(pac_correo,pac_telefono)
);

CREATE TABLE IF NOT EXISTS medico
(
	med_id INT NOT NULL AUTO_INCREMENT,
	med_nombre VARCHAR(50) NOT NULL,
    med_ap_pat VARCHAR(35) NOT NULL,
	med_ap_mat VARCHAR(35),
	med_especialidad VARCHAR(30) NOT NULL,
	med_consultorio VARCHAR(4) NOT NULL,
	med_correo VARCHAR(50),
	med_telefono CHAR(10) NOT NULL,
	med_honorarios DECIMAL(7, 2) NOT NULL COMMENT 'Pesos MXN',
    PRIMARY KEY(med_id),
    INDEX idx_nombcomp(med_ap_pat ,med_ap_mat, med_nombre),
    INDEX idx_especialidad(med_especialidad),
    UNIQUE uni_datosmed(med_telefono, med_correo, med_consultorio)
);

CREATE TABLE IF NOT EXISTS cita
(
	cit_id INT NOT NULL AUTO_INCREMENT, cit_fecha_hora DATETIME NOT NULL, cit_diagnostico TINYTEXT NOT NULL,
	cit_pac_id INT,
	cit_med_id INT,
    PRIMARY KEY (cit_id),
	INDEX idx_fechahora (cit_fecha_hora),
	UNIQUE uni_fechapac (cit_fecha_hora, cit_pac_id),
	UNIQUE uni_fechamed (cit_fecha_hora, cit_med_id),
    CONSTRAINT fk_pac_cit
		FOREIGN KEY (cit_pac_id)
        REFERENCES paciente(pac_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
	CONSTRAINT fk_med_cit
		FOREIGN KEY (cit_med_id)
        REFERENCES medico(med_id)
        ON UPDATE SET NULL
        ON DELETE SET NULL
);


# Las tablas se pueden catalogar en tres tipos:
# - Independientes: NO TIENEN llave foránea. No dependen de otras tablas para funcionar.
# - Dependientes: SI TIENEN llave(5) foránea(s). Dependen de otras tablas para funcionar.
#
# - Descriptivas: Son un tipo de tabla dependiente, que proporciona información
# adicional sobre otra tabla.
# Tenemos 3 opciones para formar la llave primaria de una tabla dependiente
# 1. Usar la combinación de llaves foráneas, SI ES ÚNICA.
# 2. Usar la combinación de llaves foráneas y UN ATRIBUTO ADICIONAL tomado de la tabla
# 3. Usar una llave primaria artificial (crear un índice propio)
	# Numérica.
    # Auto Incremental.
# Al momento de crear la base de datos
# se debe inicar con la creación de las tablas independientes
# --> Si no, me marca error al momento de crear primero una tabla dependiente

-- Insertar 10 pacientes
INSERT INTO paciente (pac_nombre, pac_ap_pat, pac_ap_mat, pac_correo, pac_telefono)
VALUES
('Juan', 'García', 'Hernández', 'j.garciahernandez@example.com', '1234567890'),
('María', 'López', 'Martínez', 'm.lopezmartinez@example.com', '2345678901'),
('Pedro', 'Martínez', 'González', 'p.martinezgonzalez@example.com', '3456789012'),
('Ana', 'Hernández', 'Pérez', 'a.hernandezperez@example.com', '4567890123'),
('Luis', 'González', 'Rodríguez', 'l.gonzalezrodriguez@example.com', '5678901234'),
('Laura', 'Rodríguez', 'Sánchez', 'l.rodriguezsanchez@example.com', '6789012345'),
('Carlos', 'Sánchez', 'Gómez', 'c.sanchezgomez@example.com', '7890123456'),
('Mónica', 'Gómez', 'Díaz', 'm.gomezdiaz@example.com', '8901234567'),
('Jorge', 'Díaz', 'Muñoz', 'j.diazmunoz@example.com', '9012345678'),
('Diana', 'Muñoz', 'Flores', 'd.munozflores@example.com', '0123456789');

-- Insertar 2 médicos para cada una de las siguientes especialidades
# general, gastroenterología, traumatología, ginecología, urología
INSERT INTO medico (med_nombre, med_ap_pat, med_ap_mat, med_especialidad, med_consultorio, med_correo, med_telefono, med_honorarios)
VALUES
('Dr. García', 'Pérez', 'Martínez', 'General', '101', 'dr.garcia@example.com', '2345678901', 1500.00),
('Dra. Martínez', 'López', 'Hernández', 'General', '102', 'dra.martinez@example.com', '3456789012', 1600.00),
('Dr. González', 'García', 'Sánchez', 'Gastroenterología', '201', 'dr.gonzalez@example.com', '4567890123', 1800.00),
('Dra. López', 'Gómez', 'Rodríguez', 'Gastroenterología', '202', 'dra.lopez@example.com', '5678901234', 1900.00),
('Dr. Pérez', 'Hernández', 'Muñoz', 'Traumatología', '301', 'dr.perez@example.com', '6789012345', 1700.00),
('Dra. Sánchez', 'Rodríguez', 'Gómez', 'Traumatología', '302', 'dra.sanchez@example.com', '7890123456', 1800.00),
('Dr. Gómez', 'Sánchez', 'Díaz', 'Ginecología', '401', 'dr.gomez@example.com', '8901234567', 2000.00),
('Dra. Díaz', 'Muñoz', 'Hernández', 'Ginecología', '402', 'dra.diaz@example.com', '9012345678', 2100.00),
('Dr. Muñoz', 'Martínez', 'Gómez', 'Urología', '501', 'dr.munoz@example.com', '0123456789', 1900.00),
('Dra. Flores', 'González', 'Pérez', 'Urología', '502', 'dra.flores@example.com', '1234567890', 2000.00);

-- Insertar 3 citas para cada paciente con los médicos anteriores
INSERT INTO cita (cit_fecha_hora, cit_diagnostico, cit_pac_id, cit_med_id)
VALUES
('2024-04-10 08:30:00', 'Dolor de cabeza', 1, 1),
('2024-04-20 10:00:00', 'Malestar estomacal', 1, 3),
('2024-05-05 14:30:00', 'Control de rutina', 1, 5),
('2024-04-15 09:45:00', 'Consulta por resfriado', 2, 2),
('2024-04-25 11:15:00', 'Dolor abdominal', 2, 4),
('2024-05-10 15:00:00', 'Seguimiento de lesión', 2, 6),
('2024-04-12 08:00:00', 'Examen ginecológico', 3, 7),
('2024-04-22 09:30:00', 'Consulta por dolor en cadera', 3, 9),
('2024-05-08 11:45:00', 'Dolor lumbar', 3, 11),
('2024-04-14 10:20:00', 'Revisión de operación previa', 4, 8),
('2024-04-24 12:00:00', 'Control postoperatorio', 4, 10),
('2024-05-06 14:15:00', 'Seguimiento de tratamiento', 4, 12),
('2024-04-13 11:00:00', 'Consulta por dolor en rodilla', 5, 9),
('2024-04-23 13:30:00', 'Control de lesión', 5, 11),
('2024-05-07 15:20:00', 'Revisión de diagnóstico previo', 5, 1),
('2024-04-16 12:30:00', 'Dolor abdominal', 6, 4),
('2024-04-26 14:45:00', 'Examen de rutina', 6, 6),
('2024-05-09 08:50:00', 'Consulta por malestar general', 6, 8),
('2024-04-17 13:00:00', 'Control de presión arterial', 7, 3),
('2024-04-27 15:10:00', 'Seguimiento de tratamiento', 7, 5),
('2024-05-11 09:00:00', 'Consulta por dolor de cabeza', 7, 7),
('2024-04-18 14:15:00', 'Control de rutina', 8, 2),
('2024-04-28 10:20:00', 'Seguimiento de lesión', 8, 4),
('2024-05-12 11:30:00', 'Dolor de espalda', 8, 6),
('2024-04-19 15:30:00', 'Revisión de diagnóstico previo', 9, 11),
('2024-04-29 08:40:00', 'Consulta por dolor de cabeza', 9, 1),
('2024-05-13 12:45:00', 'Control de presión arterial', 9, 3),
('2024-04-11 08:10:00', 'Consulta por dolor abdominal', 10, 10),
('2024-04-21 09:40:00', 'Seguimiento de tratamiento', 10, 12),
('2024-05-04 10:50:00', 'Dolor de cabeza', 10, 2);

