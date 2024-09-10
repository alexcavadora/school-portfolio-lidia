# Created: 11/03/2024
# Created by: Alejandro Alonso Sanchez
# Semester: 2024-EJ
# Course: Bases de Datos Relacionales
# Description: Crear una base de datos para contro]
# de comentarios en publicaciones de redes sociales:
DROP DATABASE IF EXISTS red_social;
CREATE DATABASE IF NOT EXISTS red_social;
USE red_social;
# Crear la tabla usuario
CREATE TABLE IF NOT EXISTS usuario(
	usu_id INT NOT NULL AUTO_INCREMENT, usu_nombre VARCHAR(50) NOT NULL,
	usu_ap_pat VARCHAR(35) NOT NULL,
	usu_ap_mat VARCHAR(35),
	usu_correo VARCHAR(40) NOT NULL,
	PRIMARY KEY (usu_id),
	INDEX idx_nomcomp (usu_ap_pat, usu_ap_mat, usu_nombre),
	UNIQUE uni_correo (usu_correo)
);
# Crear la tabla publicacion
CREATE TABLE IF NOT EXISTS publicacion(
	pub_id INT NOT NULL AUTO_INCREMENT,
    pub_texto TINYTEXT NOT NULL,
	pub_fecha_hora DATETIME NOT NULL,
	pub_estado ENUM('activa', 'revisión', 'bloqueada') 
    NOT NULL, pub_usu_id INT NOT NULL, 
    PRIMARY KEY (pub_id),
	INDEX idx_fecha (pub_fecha_hora),
	INDEX idx_edo (pub_estado),
	CONSTRAINT f_usuario_pub
	FOREIGN KEY (pub_usu_id)
	REFERENCES usuario (usu_id)
	ON UPDATE CASCADE
	ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS comentario(
com_id INT NOT NULL AUTO_INCREMENT, 
com_texto TINYTEXT NOT NULL, 
com_fecha_hora DATETIME NOT NULL, 
com_estado ENUM('activo', 'revisión', 'bloqueado') NOT NULL, 
com_usu_id INT NOT NULL, 
com_pub_id INT NOT NULL, 
PRIMARY KEY (com_id),
INDEX idx_fecha (com_fecha_hora),
INDEX idx_edo (com_estado),
CONSTRAINT fk_usu_com
	FOREIGN KEY (com_usu_id)
	REFERENCES usuario (usu_id)
	ON UPDATE CASCADE
	ON DELETE CASCADE, 
CONSTRAINT fk_pub_com
	FOREIGN KEY (com_pub_id)
	REFERENCES publicacion (pub_id)
	ON UPDATE CASCADE
	ON DELETE CASCADE
);
-- Insertar 10 usuarios
INSERT INTO usuario (usu_nombre, usu_ap_pat, usu_ap_mat, usu_correo)
VALUES
('Alejandro', 'Alonso', 'Sanchez', 'a.alonsosanchez@example.com'),
('Maria', 'Gonzalez', 'Lopez', 'm.gonzalezlopez@example.com'),
('Carlos', 'Martinez', 'Perez', 'c.martinezperez@example.com'),
('Laura', 'Rodriguez', 'Garcia', 'l.rodriguezgarcia@example.com'),
('Daniel', 'Sanchez', 'Hernandez', 'd.sanchezhernandez@example.com'),
('Ana', 'Lopez', 'Fernandez', 'a.lopezfernandez@example.com'),
('David', 'Gomez', 'Diaz', 'd.gomezdiaz@example.com'),
('Sofia', 'Perez', 'Gonzalez', 's.perezgonzalez@example.com'),
('Diego', 'Fernandez', 'Martinez', 'd.fernandezmartinez@example.com'),
('Luisa', 'Hernandez', 'Rodriguez', 'l.hernandezrodriguez@example.com');

-- Insertar 2 publicaciones para cada usuario
INSERT INTO publicacion (pub_texto, pub_fecha_hora, pub_estado, pub_usu_id)
VALUES
('Publicación 1 de Alejandro', '2024-04-01 10:00:00', 'activa', 1),
('Publicación 2 de Alejandro', '2024-04-05 15:30:00', 'activa', 1),
('Publicación 1 de Maria', '2024-04-02 11:20:00', 'activa', 2),
('Publicación 2 de Maria', '2024-04-06 16:45:00', 'activa', 2),
('Publicación 1 de Carlos', '2024-04-03 09:10:00', 'activa', 3),
('Publicación 2 de Carlos', '2024-04-07 14:00:00', 'activa', 3),
('Publicación 1 de Laura', '2024-04-04 08:00:00', 'activa', 4),
('Publicación 2 de Laura', '2024-04-08 13:20:00', 'activa', 4),
('Publicación 1 de Daniel', '2024-04-05 10:30:00', 'activa', 5),
('Publicación 2 de Daniel', '2024-04-09 12:40:00', 'activa', 5),
('Publicación 1 de Ana', '2024-04-06 12:15:00', 'activa', 6),
('Publicación 2 de Ana', '2024-04-10 11:00:00', 'activa', 6),
('Publicación 1 de David', '2024-04-07 14:45:00', 'activa', 7),
('Publicación 2 de David', '2024-04-11 10:20:00', 'activa', 7),
('Publicación 1 de Sofia', '2024-04-08 16:00:00', 'activa', 8),
('Publicación 2 de Sofia', '2024-04-12 09:30:00', 'activa', 8),
('Publicación 1 de Diego', '2024-04-09 09:40:00', 'activa', 9),
('Publicación 2 de Diego', '2024-04-13 15:10:00', 'activa', 9),
('Publicación 1 de Luisa', '2024-04-10 13:50:00', 'activa', 10),
('Publicación 2 de Luisa', '2024-04-14 08:45:00', 'activa', 10);

-- Insertar 2 comentarios para cada publicación
INSERT INTO comentario (com_texto, com_fecha_hora, com_estado, com_usu_id, com_pub_id)
VALUES
('Comentario 1 a la publicación de Alejandro', '2024-04-02 12:00:00', 'activo', 2, 1),
('Comentario 2 a la publicación de Alejandro', '2024-04-03 14:30:00', 'activo', 3, 1),
('Comentario 1 a la publicación de Maria', '2024-04-03 13:20:00', 'activo', 4, 2),
('Comentario 2 a la publicación de Maria', '2024-04-04 11:45:00', 'activo', 5, 2),
('Comentario 1 a la publicación de Carlos', '2024-04-04 10:00:00', 'activo', 6, 3),
('Comentario 2 a la publicación de Carlos', '2024-04-05 09:15:00', 'activo', 7, 3),
('Comentario 1 a la publicación de Laura', '2024-04-05 14:40:00', 'activo', 8, 4),
('Comentario 2 a la publicación de Laura', '2024-04-06 10:20:00', 'activo', 9, 4),
('Comentario 1 a la publicación de Daniel', '2024-04-06 11:30:00', 'activo', 10, 5),
('Comentario 2 a la publicación de Daniel', '2024-04-07 15:50:00', 'activo', 1, 5),
('Comentario 1 a la publicación de Ana', '2024-04-07 16:30:00', 'activo', 2, 6),
('Comentario 2 a la publicación de Ana', '2024-04-08 14:00:00', 'activo', 3, 6),
('Comentario 1 a la publicación de David', '2024-04-08 15:20:00', 'activo', 4, 7),
('Comentario 2 a la publicación de David', '2024-04-09 13:10:00', 'activo', 5, 7),
('Comentario 1 a la publicación de Sofia', '2024-04-09 11:40:00', 'activo', 6, 8),
('Comentario 2 a la publicación de Sofia', '2024-04-10 12:25:00', 'activo', 7, 8),
('Comentario 1 a la publicación de Diego', '2024-04-10 16:00:00', 'activo', 8, 9),
('Comentario 2 a la publicación de Diego', '2024-04-11 08:50:00', 'activo', 9, 9),
('Comentario 1 a la publicación de Luisa', '2024-04-11 10:30:00', 'activo', 10, 10),
('Comentario 2 a la publicación de Luisa', '2024-04-12 14:15:00', 'activo', 1, 10);
