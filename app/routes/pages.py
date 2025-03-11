from flask import render_template
from app import app

@app.route('/educational')
def educational():
    return render_template('educational.html')

@app.route('/technical')
def technical():
    return render_template('technical.html') 