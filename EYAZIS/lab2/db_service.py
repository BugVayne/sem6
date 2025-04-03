import os
import re
import psycopg2

def sanitize_filename(title):
    return re.sub(r'[<>:"/\\|?*]', '_', title)

def get_db_connection():
    conn = psycopg2.connect(
        dbname='words_articles',
        user='postgres',
        password='1234',
        host='localhost',
        port='5432'
    )
    return conn


def read_articles():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, source_url, authors, publish_date FROM articles;")
    articles = cursor.fetchall()
    cursor.close()
    conn.close()
    return articles


def read_words():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT w.word, SUM(w.count) as total_count
        FROM words w
        GROUP BY w.word
        ORDER BY w.word;  -- Order alphabetically
    """)
    words = cursor.fetchall()
    cursor.close()
    conn.close()
    return words

def get_word_details_by_id(word_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT article_id, word, pos_info, count FROM words WHERE id = %s;", (word_id,))
    word_details = cursor.fetchone()
    cursor.close()
    conn.close()
    return word_details



def get_word_details(word):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.name, w.word, w.count, w.pos_info, 
               a.authors, a.publish_date, a.source_url
        FROM words w
        JOIN articles a ON w.article_id = a.id
        WHERE w.word = %s
        ORDER BY a.name;
    """, (word,))
    word_details = cursor.fetchall()
    cursor.close()
    conn.close()

    articles_directory = 'scrapper/articles'

    detailed_data = []
    for detail in word_details:
        article_name = detail[0]
        article_name_sanitized = sanitize_filename(article_name)

        word = detail[1]

        authors = detail[4]
        publish_date = detail[5]
        source_url = detail[6]

        article_file_path = os.path.join(articles_directory, f"{article_name_sanitized}.txt")

        sentences = []
        if os.path.exists(article_file_path):
            with open(article_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

                article_text = ""
                capture_text = False

                for line in lines:
                    if capture_text:
                        article_text += line.strip() + " "  # Append the article text
                    elif line.strip() == "":  # Blank line indicates end of metadata
                        capture_text = True

                sentence_list = re.split(r'(?<=[.!?])\s+', article_text)

                for sentence in sentence_list:
                    if re.search(rf'\b{word}\b', sentence, re.IGNORECASE):
                        sentences.append(sentence.strip())

        detailed_data.append({
            'article_name': article_name,
            'word': word,
            'count': detail[2],
            'pos_info': detail[3],
            'authors': authors,
            'publish_date': publish_date,
            'source_url': source_url,
            'sentences': sentences
        })

    return detailed_data