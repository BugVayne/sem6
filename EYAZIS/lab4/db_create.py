import psycopg2

# Подключение к базе данных
conn = psycopg2.connect(
    dbname='semantic_analysis',
    user='postgres',
    password='1234',
    host='localhost',
    port='5432'
)

# Создание курсора
cur = conn.cursor()

# Создание таблиц с новыми колонками
cur.execute('''
CREATE TABLE IF NOT EXISTS sentences (
    id SERIAL PRIMARY KEY,
    sentence TEXT NOT NULL,
    tone VARCHAR(50), 
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
    head TEXT NOT NULL,
    definitions TEXT[], 
    synonyms TEXT[],    
    antonyms TEXT[]    
);
''')

# Сохранение изменений
conn.commit()

# Закрытие курсора и соединения
cur.close()
conn.close()