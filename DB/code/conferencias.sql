# CREATED: 15/02/2024
# AUTHOR: Alejandro Alonso Sánchez
# SEMESTER: 2024-Ene-Jun
# COURSE: Bases de datos relacionales.
# DESCRIPTION: Crear una base de datos para la gestión de asistencia de una conferencia
DROP DATABASE IF EXISTS escuela_eventos;
CREATE DATABASE IF NOT EXISTS escuela_eventos;
USE escuela_eventos;
CREATE TABLE IF NOT EXISTS estudiante
(
	est_nua INT NOT NULL AUTO_INCREMENT,
    est_nom VARCHAR(30) NOT NULL,
    est_ap_pat VARCHAR(30) NOT NULL,
    est_ap_mat VARCHAR(30),
    est_correo VARCHAR(30),
    est_carrera VARCHAR(40),
    PRIMARY KEY(est_nua),
    INDEX idx_est_nom(est_ap_pat,est_nom,  est_ap_mat),
    INDEX idx_est_correo(est_correo),
    UNIQUE uni_est_correo(est_correo)
);

CREATE TABLE IF NOT EXISTS evento
(
	eve_id INT NOT NULL AUTO_INCREMENT,
    eve_fecha DATE NOT NULL,
    eve_hora TIME NOT NULL,
    eve_lugar ENUM('A101', 'A102', 'AUDIOVISUAL'),
    eve_duracion INT COMMENT 'MINUTOS',
    eve_ponente VARCHAR(90) NOT NULL,
    PRIMARY KEY (eve_id),
    INDEX idx_fecha_hora(eve_fecha,eve_hora),
    UNIQUE uni_eve_fecha_lugar(eve_fecha, eve_hora, eve_lugar)    
);

CREATE TABLE IF NOT EXISTS asistencia
(
	asi_est_nua INT NOT NULL,
    asi_eve_id INT NOT NULL,
    asi_hora_llegada TIME,
    PRIMARY KEY(asi_est_nua, asi_eve_id),
    CONSTRAINT fk_asi_est_id
		FOREIGN KEY (asi_est_nua)
        REFERENCES estudiante(est_nua)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
	CONSTRAINT fk_asi_eve_id
		FOREIGN KEY (asi_eve_id)
        REFERENCES evento(eve_id)
        ON DELETE RESTRICT
        ON UPDATE RESTRICT
);

# 1. Dentro de la tabla alumnos agrega un atributo para almacenar el numero de teléfono de los alumnos; agrégalo al final de la tabla. 
#ALTER TABLE estudiante ADD est_telefono CHAR(10);

# 2. En la tabla estudiante elimina el índice para el atributo est_correo.
#ALTER TABLE estudiante DROP INDEX idx_est_correo;

# 3. En la tabla de evento elimina el atributo eve_duracion. 
#ALTER TABLE evento DROP eve_duracion;

# 4. En la tabla de evento agrega el atributo eve_hora_fin después del atributo eve_hora, con un tipo de dato TIME NOT NULL.
#ALTER TABLE evento ADD eve_hora_fin TIME NOT NULL AFTER eve_hora;

# 5. En la tabla evento cambie el nombre del atributo eve_hora por el nombre eve_hora_inicio.
#ALTER TABLE evento RENAME COLUMN eve_hora TO eve_hora_inicio;

-- Insertar 10 estudiantes de diferentes carreras
INSERT INTO estudiante (est_nom, est_ap_pat, est_ap_mat, est_correo, est_carrera) 
VALUES 
('Juan', 'García', 'Hernández', 'j.garciahernandez@ugto.mx', 'Ingeniería en Sistemas Computacionales'),
('María', 'López', 'Martínez', 'm.lopezmartinez@ugto.mx', 'Ingeniería Industrial'),
('Pedro', 'Martínez', 'González', 'p.martinezgonzalez@ugto.mx', 'Administración de Empresas'),
('Ana', 'Hernández', 'Pérez', 'a.hernandezperez@ugto.mx', 'Derecho'),
('Luis', 'González', 'Rodríguez', 'l.gonzalezrodriguez@ugto.mx', 'Medicina'),
('Laura', 'Rodríguez', 'Sánchez', 'l.rodriguezsanchez@ugto.mx', 'Contabilidad'),
('Carlos', 'Sánchez', 'Gómez', 'c.sanchezgomez@ugto.mx', 'Arquitectura'),
('Mónica', 'Gómez', 'Díaz', 'm.gomezdiaz@ugto.mx', 'Psicología'),
('Jorge', 'Díaz', 'Muñoz', 'j.diazmunoz@ugto.mx', 'Ciencias de la Comunicación'),
('Diana', 'Muñoz', 'Flores', 'd.munozflores@ugto.mx', 'Biología');

-- Insertar 4 eventos por cada uno de los siguientes meses: abril, mayo, junio
INSERT INTO evento (eve_fecha, eve_hora, eve_lugar, eve_duracion, eve_ponente) 
VALUES 
('2024-04-10', '09:00:00', 'A101', 120, 'Dr. García'),
('2024-04-15', '14:00:00', 'A102', 90, 'Dra. Martínez'),
('2024-04-20', '10:30:00', 'AUDIOVISUAL', 150, 'Lic. Pérez'),
('2024-04-25', '11:00:00', 'A101', 120, 'Mtro. Rodríguez'),
('2024-05-05', '09:00:00', 'A102', 90, 'Dra. Sánchez'),
('2024-05-10', '14:00:00', 'AUDIOVISUAL', 150, 'Dr. Gómez'),
('2024-05-15', '10:30:00', 'A101', 120, 'Mtro. Díaz'),
('2024-05-20', '11:00:00', 'A102', 90, 'Lic. Muñoz'),
('2024-06-05', '09:00:00', 'AUDIOVISUAL', 150, 'Dra. Flores'),
('2024-06-10', '14:00:00', 'A101', 120, 'Dr. García'),
('2024-06-15', '10:30:00', 'A102', 90, 'Dra. Martínez'),
('2024-06-20', '11:00:00', 'AUDIOVISUAL', 150, 'Lic. Pérez');

-- Insertar 2 asistencias de cada estudiante a los eventos anteriores
INSERT INTO asistencia (asi_est_nua, asi_eve_id, asi_hora_llegada) 
VALUES 
(1, 1, '08:50:00'),
(1, 3, '10:25:00'),
(2, 2, '13:50:00'),
(2, 4, '10:55:00'),
(3, 5, '08:45:00'),
(3, 7, '10:20:00'),
(4, 6, '13:55:00'),
(4, 8, '10:30:00'),
(5, 9, '08:55:00'),
(5, 11, '10:35:00'),
(6, 10, '13:45:00'),
(6, 12, '10:40:00'),
(7, 1, '09:05:00'),
(7, 3, '10:30:00'),
(8, 2, '13:55:00'),
(8, 4, '11:00:00'),
(9, 5, '09:15:00'),
(9, 7, '10:40:00'),
(10, 6, '14:00:00'),
(10, 8, '10:45:00');




