from app import create_app
from app.database.db import init_db

app = create_app()

if __name__ == '__main__':
    init_db()  # Veritabanını oluştur
    app.run(debug=True) 