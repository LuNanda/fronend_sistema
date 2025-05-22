from flask import Flask, request, session, redirect, url_for
from database import get_user  # Asegúrate de implementar esta función en database.py

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta'  # Cambia esto por una clave secreta más segura

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username, password)
        
        if user:
            session['user_type'] = user['type']
            return redirect(url_for('dashboard'))  # Redirige a la página de dashboard
        else:
            return "Credenciales incorrectas", 401

    return '''
        <form method="post">
            Usuario: <input type="text" name="username"><br>
            Contraseña: <input type="password" name="password"><br>
            <input type="submit" value="Login">
        </form>
    '''

@app.route('/dashboard')
def dashboard():
    if 'user_type' not in session:
        return redirect(url_for('login'))

    user_type = session['user_type']
    if user_type == 'Director':
        return "Bienvenido, Director. Tienes acceso total."
    elif user_type == 'Miembro':
        return "Bienvenido, Miembro. Tienes acceso amplio."
    elif user_type == 'Publico':
        return "Bienvenido, Público General. Tienes acceso limitado."
    else:
        return "Tipo de usuario no reconocido."
