from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(__name__)

# Initialize the database
def init_db():
    conn = sqlite3.connect('skills.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            skill TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        skill = request.form['skill']

        if name and skill:
            # Save data to the database
            conn = sqlite3.connect('skills.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO skills (name, skill) VALUES (?, ?)', (name, skill))
            conn.commit()
            conn.close()

            return redirect(url_for('index'))

    # Retrieve all skills from the database
    conn = sqlite3.connect('skills.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, skill FROM skills')
    skills = cursor.fetchall()
    conn.close()

    return render_template('index.html', skills=skills)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
