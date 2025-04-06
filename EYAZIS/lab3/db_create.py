import psycopg2
import spacy
from spacy import displacy

# Подключение к базе данных
conn = psycopg2.connect(
    dbname='structural_analysis',
    user='postgres',
    password='1234',
    host='localhost',
    port='5432'
)

# Создание курсора
cur = conn.cursor()

# Создание таблиц
cur.execute('''
CREATE TABLE IF NOT EXISTS sentences (
    id SERIAL PRIMARY KEY,
    sentence TEXT NOT NULL,
    image BYTEA
);
''')

cur.execute('''
CREATE TABLE IF NOT EXISTS words (
    id SERIAL PRIMARY KEY,
    sentence_id INTEGER REFERENCES sentences(id),
    word TEXT NOT NULL,
    pos TEXT NOT NULL,
    dep TEXT NOT NULL,
    head TEXT NOT NULL
);
''')

# Сохранение изменений
conn.commit()

# Закрытие курсора и соединения
cur.close()
conn.close()