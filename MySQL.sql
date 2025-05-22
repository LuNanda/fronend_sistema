CREATE TABLE usuarios (
    username VARCHAR(50) PRIMARY KEY,
    password VARCHAR(255),  -- Asegúrate de usar un tamaño adecuado para el hash de la contraseña
    type ENUM('Director', 'Miembro', 'Publico') NOT NULL
);