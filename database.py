import mysql.connector
from mysql.connector import Error

def get_user(username, password):
    try:
        # Conectar a la base de datos
        connection = mysql.connector.connect(
            host='tu_host',  # Cambia esto por tu host de MySQL
            database='tu_base_de_datos',  # Cambia esto por el nombre de tu base de datos
            user='tu_usuario',  # Cambia esto por tu usuario de MySQL
            password='tu_contraseña'  # Cambia esto por tu contraseña de MySQL
        )

        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            # Consulta para obtener el usuario
            cursor.execute("SELECT * FROM usuarios WHERE username=%s AND password=%s", (username, password))
            user = cursor.fetchone()

            return user

    except Error as e:
        print(f"Error al conectarse a MySQL: {e}")
        return None

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
