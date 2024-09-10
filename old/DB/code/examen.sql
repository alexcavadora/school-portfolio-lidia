# Examen: práctico 1
# Fecha: 29-04-2024
# Carrera: Licenciatura en Ingeniería de Datos e Inteligencia Artificial
# Semestre: 2024 Enero-Junio
# Curso: Bases de Datos Relacionales
# Autor: Gómez Carranza Juan Carlos
# Descripción: Construcción de una base de datos encargada del control
# simplificado del registro de dietas para pacientes.

CREATE DATABASE IF NOT EXISTS dieta;

USE dieta;

# Primero implementamos tablas independientes
CREATE TABLE IF NOT EXISTS paciente(
	pac_id INT NOT NULL AUTO_INCREMENT, # llave primaria artificial
	pac_nombre VARCHAR(30) NOT NULL,
    pac_ap_pat VARCHAR(30) NOT NULL,
    pac_ap_mat VARCHAR(30),
    pac_fecha_nac DATE NOT NULL,
    pac_estatura DECIMAL(3, 2) NOT NULL COMMENT 'En metros',
    pac_direccion VARCHAR(60) NOT NULL,
    pac_correo VARCHAR(30) NOT NULL,
    PRIMARY KEY (pac_id),
    INDEX idx_nomcomp (pac_ap_pat, pac_ap_mat, pac_nombre),
    INDEX idx_fecha (pac_fecha_nac),
    UNIQUE uni_correo (pac_correo)
);

CREATE TABLE IF NOT EXISTS alimento(
	ali_id INT NOT NULL AUTO_INCREMENT,
	ali_nombre VARCHAR(40) NOT NULL,
    ali_medida_porcion VARCHAR(20) NOT NULL,
    ali_tipo ENUM('frutas', 'verduras', 'carnes y proteínas', 'lácteos', 'granos y cereales', 'aceites y grasas') NOT NULL,
    ali_calorias_por_porcion INT NOT NULL COMMENT 'En kilocalorias',
    ali_grasa_por_porcion INT NOT NULL COMMENT 'En gramos',
    ali_proteina_por_porcion INT NOT NULL COMMENT 'En gramos',
    ali_carbos_por_porcion INT NOT NULL COMMENT 'En gramos',
    PRIMARY KEY (ali_id),
    INDEX idx_porcion (ali_medida_porcion),
    INDEX idx_tipo (ali_tipo)
);

# Tablas dependientes
CREATE TABLE IF NOT EXISTS evo_paciente(
	evo_id INT NOT NULL AUTO_INCREMENT,
	evo_fecha DATE NOT NULL,
    evo_peso DECIMAL(4, 1) NOT NULL COMMENT 'En kilogramos',
    evo_imc DECIMAL(3, 1) NOT NULL COMMENT 'En kg/m**2',
    evo_grasa DECIMAL(3, 1) NOT NULL COMMENT 'En porcentaje',
    evo_agua DECIMAL(3, 1) NOT NULL COMMENT 'En porcentaje',
    evo_pac_id INT NOT NULL,
    PRIMARY KEY (evo_id),
    INDEX idx_fecha (evo_fecha),
    CONSTRAINT fk_pac_evo
		FOREIGN KEY (evo_pac_id)
        REFERENCES paciente (pac_id)
        ON UPDATE CASCADE 
		ON DELETE CASCADE
        # Decidí esto porque si se borra un paciente
        # no quiero guardar sus registros en esta tabla
);

CREATE TABLE IF NOT EXISTS dieta(
	die_id INT NOT NULL AUTO_INCREMENT,
	die_fecha_ini DATE NOT NULL,
    die_fecha_fin DATE NOT NULL,
    die_total INT,
    die_pac_id INT NOT NULL,
    PRIMARY KEY (die_id),
    INDEX idx_fecha_ini (die_fecha_ini),
    CONSTRAINT fk_pac_die
		FOREIGN KEY (die_pac_id)
		REFERENCES paciente (pac_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
        # Decidí esto porque si se borra un paciente
        # no quiero guardar sus registros de sus dietas.        
);

CREATE TABLE IF NOT EXISTS detalle_dieta(
	det_id INT NOT NULL AUTO_INCREMENT,
	det_tipo_comida ENUM ('desayuno', 'comida', 'cena', 'colación') NOT NULL,
	det_num_porciones INT NOT NULL,
    det_subtotal INT,
    det_die_id INT NOT NULL,
    det_ali_id INT NOT NULL,
    PRIMARY KEY (det_id),
    INDEX idx_tipo (det_tipo_comida),
    CONSTRAINT fk_die_det
		FOREIGN KEY (det_die_id)
        REFERENCES dieta (die_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
        # Decidí esto porque si se borra una dieta
        # no quiero guardar los detalles asociados a ella.
    CONSTRAINT fk_ali_det
		FOREIGN KEY (det_ali_id)
        REFERENCES alimento (ali_id)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT
        # Decidí esto porque si se borra un alimento
        # no eliminar las dietas asociados con él.
);

INSERT INTO paciente (pac_nombre, pac_ap_pat, pac_ap_mat, pac_fecha_nac, pac_estatura, pac_direccion, pac_correo)
	VALUES ('Juan', 'López', 'F', '1975-06-20', 1.75, 'ABC', 'df@ugto.mx'),
			('ABC', 'asd', 'RT', '1985-07-02', 1.60, 'DFE', 'df@gmail.com'),
            ('DEF', 'aasda', 'ad', '1990-06-02', 1.70, 'asda', 'yuy@gmail.com');

INSERT INTO alimento (ali_nombre, ali_medida_porcion, ali_tipo, ali_calorias_por_porcion, ali_grasa_por_porcion, ali_proteina_por_porcion, ali_carbos_por_porcion)    
	VALUES ('Fresa', '1 taza', 'frutas', 50, 1, 2, 3),
			('Platano', '1 taza', 'frutas', 50, 1, 2, 3),
            ('Carne', '100 gramos', 'carnes y proteínas', 200, 1, 2, 3);

INSERT INTO evo_paciente(evo_fecha, evo_peso, evo_imc, evo_grasa, evo_agua, evo_pac_id)
	VALUES	('2024-04-01', 80, 24, 40, 35, 1),
			('2024-04-02', 60, 23, 40, 35, 2),
            ('2024-04-02', 90, 30, 40, 35, 3),
            ('2024-05-01', 78, 24, 40, 35, 1);

INSERT INTO dieta (die_fecha_ini, die_fecha_fin, die_pac_id)
	VALUES ('2024-04-01', '2024-05-01', 1),
			('2024-04-02', '2024-06-01', 2),
            ('2024-04-02', '2024-05-02', 3);

           
INSERT INTO detalle_dieta(det_tipo_comida, det_num_porciones, det_die_id, det_ali_id)
	VALUES	('desayuno', 2, 1, 1),
			('cena', 1, 1, 2),
            ('desayuno', 2, 2, 1);