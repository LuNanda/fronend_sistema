import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash  # Para hashear contraseñas

def add_user(username, password, user_type):
    try:
        # Conectar a la base de datos
        connection = mysql.connector.connect(
            host='tu_host',  # Cambia esto por tu host de MySQL
            database='tu_base_de_datos',  # Cambia esto por el nombre de tu base de datos
            user='tu_usuario',  # Cambia esto por tu usuario de MySQL
            password='tu_contraseña'  # Cambia esto por tu contraseña de MySQL
        )

        if connection.is_connected():
            cursor = connection.cursor()
            # Hashear la contraseña
            hashed_password = generate_password_hash(password, method='sha256')
            # Consulta para insertar un nuevo usuario
            cursor.execute("INSERT INTO usuarios (username, password, type) VALUES (%s, %s, %s)", (username, hashed_password, user_type))
            connection.commit()  # Guardar los cambios
            print(f"Usuario '{username}' agregado exitosamente como '{user_type}'.")

    except Error as e:
        print(f"Error al conectarse a MySQL: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Ejemplo de uso
if __name__ == "__main__":
    # Cambia estos valores según sea necesario
    add_user('director@gmail.com', 'contraseña_director', 'Director')
    add_user('miembro@gmail.com', 'contraseña_miembro', 'Miembro')
    add_user('publico@gmail.com', 'contraseña_publico', 'Publico')
