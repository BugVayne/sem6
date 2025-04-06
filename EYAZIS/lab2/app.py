import os
import json
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, make_response
import db_service as db
from lab2 import db_load
from datetime import date


app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/articles/<int:article_id>/export-json', methods=['GET'])
def export_article_to_json(article_id):
    conn = db.get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, source_url, authors, publish_date FROM articles WHERE id = %s;", (article_id,))
    article = cursor.fetchone()

    if not article:
        flash("Article not found", "danger")
        return redirect(url_for('articles'))

    # Convert the article data to a dictionary
    article_dict = {
        'id': article[0],
        'name': article[1],
        'source_url': article[2],
        'authors': article[3],
        # Convert the date object to a string using .isoformat()
        'publish_date': article[4].isoformat() if isinstance(article[4], date) else article[4]
    }

    cursor.execute("""
        SELECT word, count, pos_info 
        FROM words 
        WHERE article_id = %s 
        ORDER BY word;
    """, (article_id,))
    words = cursor.fetchall()

    # Convert word tuples into dictionaries
    words_list = [
        {
            'word': word[0],
            'count': word[1],
            'pos_info': word[2]
        }
        for word in words
    ]

    cursor.close()
    conn.close()

    # Combine article and words into a single JSON object
    data = {
        'article': article_dict,
        'words': words_list
    }

    # Generate the JSON response
    response = make_response(json.dumps(data, indent=4))
    response.headers['Content-Type'] = 'application/json'
    response.headers['Content-Disposition'] = f'attachment; filename=article_{article_id}.json'
    return response

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/articles', methods=['GET', 'POST'])
def articles():
    if request.method == 'POST':
        name = request.form['name']
        source_url = request.form['source_url']
        authors = request.form['authors']
        publish_date = request.form['publish_date']
        db.create_article(name, source_url, authors, publish_date)
        return redirect(url_for('articles'))

    articles_data = db.read_articles()
    return render_template('articles.html', articles=articles_data)


@app.route('/articles/delete/<int:article_id>')
def delete_article(article_id):
    db.delete_article(article_id)
    return redirect(url_for('articles'))


@app.route('/words')
def words():
    search = request.args.get('search', '').strip().lower()
    pos = request.args.get('pos', '').strip().upper()

    conn = db.get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT word, SUM(count) AS total_count
        FROM words
        WHERE 1 = 1
    """
    params = []

    if search:
        query += " AND LOWER(word) LIKE %s"
        params.append(f"%{search}%")

    if pos:
        query += " AND pos_info ILIKE %s"
        params.append(f"POS: {pos},%")

    query += " GROUP BY word ORDER BY word"

    cursor.execute(query, tuple(params))
    words = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('words.html', words=words)
@app.route('/words/delete/<int:word_id>')
def delete_word(word_id):
    db.delete_word(word_id)
    return redirect(url_for('words'))


@app.route('/articles/<int:article_id>')
def article_detail(article_id):
    conn = db.get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, source_url, authors, publish_date FROM articles WHERE id = %s;", (article_id,))
    article = cursor.fetchone()

    if not article:
        flash("Article not found", "danger")
        return redirect(url_for('articles'))

    article_dict = {
        'id': article[0],
        'name': article[1],
        'source_url': article[2],
        'authors': article[3],
        'publish_date': article[4]
    }

    cursor.execute("""
        SELECT word, count, pos_info 
        FROM words 
        WHERE article_id = %s 
        ORDER BY word;
    """, (article_id,))
    words = cursor.fetchall()

    words_list = []
    for word in words:
        words_list.append({
            'word': word[0],
            'count': word[1],
            'pos_info': word[2]
        })

    cursor.close()
    conn.close()

    return render_template('article_detail.html', article=article_dict, words=words_list)

@app.route('/update-pos-info', methods=['POST'])
def update_pos_info():
    article_name = request.form.get('article_name')
    word = request.form.get('word')
    new_pos_info = request.form.get('pos_info')

    if not article_name or not word or not new_pos_info:
        flash("Invalid input. Please provide all required fields.", "danger")
        return redirect(request.referrer)

    try:
        conn = db.get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE words
            SET pos_info = %s
            WHERE word = %s AND article_id = (
                SELECT id FROM articles WHERE name = %s
            );
        """, (new_pos_info, word, article_name))

        conn.commit()
        cursor.close()
        conn.close()

        flash(f"Successfully updated part of speech for '{word}' in '{article_name}'.", "success")
    except Exception as e:
        # Handle database errors
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(request.referrer)

    return redirect(request.referrer)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash("No file part in the request.")
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash("No file selected.")
        return redirect(url_for('index'))

    if file and file.filename.endswith('.txt'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            db_load.process_file_and_save(file_path)
            flash(f"File '{file.filename}' successfully processed!")
        except Exception as e:
            flash(f"An error occurred while processing the file: {str(e)}")
    else:
        flash("Invalid file type. Please upload a .txt file.")

    return redirect(url_for('index'))


@app.route('/words/<string:word>')
def word_details(word):
    details = db.get_word_details(word)
    return render_template('word_details.html', word=word, details=details)

@app.route('/user-guide')
def user_guide():
    return render_template('user_guide.html')

if __name__ == '__main__':
    app.run(debug=True)