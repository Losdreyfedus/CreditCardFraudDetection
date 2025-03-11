from flask import Flask
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Route'ları yükle
from app.routes import main, api, pages

def create_app():
    return app