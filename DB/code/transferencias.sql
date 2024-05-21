# CREATED: 15/02/2024
# AUTHOR: Alejandro Alonso Sánchez
# SEMESTER: 2024-Ene-Jun
# COURSE: Bases de datos relacionales.
# DESCRIPTION: Crear una base de datos para la gestión de manejo de bancos
DROP DATABASE IF EXISTS transferencias;
CREATE DATABASE IF NOT EXISTS transferencias;
USE transferencias;

CREATE TABLE IF NOT EXISTS cliente
(
	cli_id INT AUTO_INCREMENT NOT NULL,
	cli_nombre VARCHAR(30) NOT NULL,
	cli_ap_pat VARCHAR(30) NOT NULL,
	cli_ap_mat VARCHAR (30),
	cli_fecha_nac DATE NOT NULL,
    cli_direccion VARCHAR(80) NOT NULL,
	cli_telefono CHAR(10) NOT NULL,
	cli_correo VARCHAR(40) NOT NULL,
	cli_rfc VARCHAR (13) COMMENT 'Con homoclave', 
    PRIMARY KEY(cli_id),
    INDEX idx_nom(cli_ap_pat, cli_ap_mat, cli_nombre),
    UNIQUE uni_tel(cli_telefono),
    UNIQUE uni_correo(cli_correo)
    );
 
CREATE TABLE IF NOT EXISTS cuenta
(
	cue_id INT NOT NULL AUTO_INCREMENT,
	cue_numero VARCHAR(18) NOT NULL,
	cue_clabe CHAR(18) NOT NULL,
    cue_fecha_apertura DATE NOT NULL,
    cue_tipo ENUM('Corriente', 'Chequera', 'Ahorro', 'Nómina', 'Empresarial','Dólares'), #Tipos de cuentas en BBVA
    cue_saldo DECIMAL (12,2),
    cue_cliente INT,
    PRIMARY KEY (cue_id),
    UNIQUE uni_num(cue_numero),
    UNIQUE uni_cla(cue_clabe),
    CONSTRAINT fk_cli_cue
		FOREIGN KEY (cue_cliente)
        REFERENCES cliente(cli_id)
		ON UPDATE CASCADE
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS transferencias
(
	tra_id INT NOT NULL AUTO_INCREMENT,
	tra_monto DECIMAL(20,2) COMMENT 'MXN PESO' NOT NULL,
	tra_fecha_hora DATETIME NOT NULL,
	tra_concepto VARCHAR(20),
    tra_referencia VARCHAR(20),
	tra_medio ENUM('App', 'Cajero', 'Internet', 'Ventanilla'),
    tra_frm_cue_id INT,
    tra_to_cue_id INT,
    PRIMARY KEY (tra_id),
    CONSTRAINT fk_desde_cue_cla
		FOREIGN KEY (tra_frm_cue_id)
		REFERENCES cuenta (cue_id)
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT fk_hacia_cue_cla
		FOREIGN KEY (tra_to_cue_id)
		REFERENCES cuenta(cue_id)
		ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    UNIQUE uni_tra_frm_to_hora(tra_fecha_hora, tra_frm_cue_id, tra_to_cue_id)
);

-- Insertar 10 clientes
INSERT INTO cliente (cli_nombre, cli_ap_pat, cli_ap_mat, cli_fecha_nac, cli_direccion, cli_telefono, cli_correo, cli_rfc)
VALUES
('Juan', 'García', 'Hernández', '1980-05-10', 'Calle 123, Colonia Centro', '1234567890', 'j.garciahernandez@example.com', 'GHJ801205XXX'),
('María', 'López', 'Martínez', '1990-09-15', 'Av. Principal 456, Colonia Juárez', '2345678901', 'm.lopezmartinez@example.com', 'LMM900915XXX'),
('Pedro', 'Martínez', 'González', '1985-03-20', 'Paseo de las Flores 789, Colonia Libertad', '3456789012', 'p.martinezgonzalez@example.com', 'PMG850320XXX'),
('Ana', 'Hernández', 'Pérez', '1988-07-05', 'Avenida Revolución 321, Colonia Reforma', '4567890123', 'a.hernandezperez@example.com', 'AHP880705XXX'),
('Luis', 'González', 'Rodríguez', '1995-11-30', 'Boulevard de la Luna 654, Colonia Luna Llena', '5678901234', 'l.gonzalezrodriguez@example.com', 'LGR951130XXX'),
('Laura', 'Rodríguez', 'Sánchez', '1982-12-15', 'Calle del Sol 987, Colonia Sol Naciente', '6789012345', 'l.rodriguezsanchez@example.com', 'LRS821215XXX'),
('Carlos', 'Sánchez', 'Gómez', '1987-04-25', 'Plaza del Río 741, Colonia Río Bravo', '7890123456', 'c.sanchezgomez@example.com', 'CSG870425XXX'),
('Mónica', 'Gómez', 'Díaz', '1993-08-20', 'Avenida de la Paz 369, Colonia Paz y Amor', '8901234567', 'm.gomezdiaz@example.com', 'MGD930820XXX'),
('Jorge', 'Díaz', 'Muñoz', '1989-10-05', 'Calle de la Esperanza 852, Colonia Esperanza', '9012345678', 'j.diazmunoz@example.com', 'JDM891005XXX'),
('Diana', 'Muñoz', 'Flores', '1987-04-25', 'Avenida Principal 1234, Colonia Principal', '0123456789', 'd.munozflores@example.com', 'DMF870425XXX');

-- Insertar 1 cuenta personal para cada cliente
INSERT INTO cuenta (cue_numero, cue_clabe, cue_fecha_apertura, cue_tipo, cue_saldo, cue_cliente)
VALUES
('123456789012345678', '456789012345678912', '2020-01-01', 'Ahorro', 5000.00, 1),
('234567890123456789', '567890123456789012', '2020-02-01', 'Ahorro', 7000.00, 2),
('345678901234567890', '678901234567890123', '2020-03-01', 'Ahorro', 6000.00, 3),
('456789012345678901', '789012345678901234', '2020-04-01', 'Ahorro', 8000.00, 4),
('567890123456789012', '890123456789012345', '2020-05-01', 'Ahorro', 5500.00, 5),
('678901234567890123', '901234567890123456', '2020-06-01', 'Ahorro', 9000.00, 6),
('789012345678901234', '012345678901234567', '2020-07-01', 'Ahorro', 4000.00, 7),
('890123456789012345', '123456789012345678', '2020-08-01', 'Ahorro', 7500.00, 8),
('901234567890123456', '234567890123456789', '2020-09-01', 'Ahorro', 6500.00, 9),
('012345678901234567', '345678901234567890', '2020-10-01', 'Ahorro', 8500.00, 10);

-- Insertar 5 cuentas empresariales para 5 clientes
INSERT INTO cuenta (cue_numero, cue_clabe, cue_fecha_apertura, cue_tipo, cue_saldo, cue_cliente)
VALUES
('112233445566778899', '223344556677889911', '2020-01-01', 'Empresarial', 15000.00, 1),
('223344556677889911', '334455667788991122', '2020-02-01', 'Empresarial', 18000.00, 2),
('334455667788991122', '445566778899112233', '2020-03-01', 'Empresarial', 12000.00, 3),
('445566778899112233', '556677889911223344', '2020-04-01', 'Empresarial', 20000.00, 4),
('556677889911223344', '667788991122334455', '2020-05-01', 'Empresarial', 16000.00, 5);

-- Insertar 5 transferencias entre clientes
INSERT INTO transferencias (tra_monto, tra_fecha_hora, tra_concepto, tra_referencia, tra_medio, tra_frm_cue_id, tra_to_cue_id)
VALUES
(1500.00, '2024-04-01 10:00:00', 'Pago de servicios', 'Ref1234', 'Internet', 1, 2),
(2000.00, '2024-05-05 12:30:00', 'Pago de compra', 'Ref5678', 'App', 3, 4),
(3000.00, '2024-06-10 15:45:00', 'Transferenciafam', 'Ref91011', 'Cajero', 5, 6),
(2500.00, '2024-07-15 09:20:00', 'Pago de deuda', 'Ref121314', 'Ventanilla', 7, 8),
(1800.00, '2024-08-20 14:10:00', 'Compra en línea', 'Ref151617', 'Internet', 9, 10);
