from flask import Flask, render_template, request, redirect, flash, url_for
from db_service import fetch_sentences, save_sentences_to_db, fetch_sentence_details, update_word_in_db, \
    extract_sentences_from_rtf
import os


app = Flask(__name__)
app.secret_key = "secret_key"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sentences')
def show_sentences():
    sentences = fetch_sentences()  # Fetch sentences from the database
    return render_template('sentences.html', sentences=sentences)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and file.filename.endswith('.rtf'):
        # Save the file to a temporary location
        temp_file_path = os.path.join('uploads', file.filename)  # Ensure the 'uploads' directory exists
        file.save(temp_file_path)

        # Extract sentences from RTF file
        sentences = extract_sentences_from_rtf(temp_file_path)

        # Save sentences to the database
        save_sentences_to_db(sentences)

        # Optionally: Delete the temporary file after processing
        os.remove(temp_file_path)

        flash('File successfully uploaded and sentences saved!')
        return redirect('/')  # Redirect to the sentences page after upload
    else:
        flash('Invalid file format. Please upload an RTF file.')
        return redirect(request.url)

@app.route('/sentence/<int:sentence_id>/update_word', methods=['POST'])
def update_word(sentence_id):
    word = request.form.get('word')
    pos = request.form.get('pos')
    dep = request.form.get('dep')
    head = request.form.get('head')
    definitions = request.form.getlist('definitions')  # Expecting a list of definitions
    synonyms = request.form.getlist('synonyms')        # Expecting a list of synonyms
    antonyms = request.form.getlist('antonyms')        # Expecting a list of antonyms

    # Ensure the word is provided (word is used as a key to identify the record)
    if not word or not pos or not dep or not head:
        flash('field is required.', 'danger')
        return redirect(url_for('sentence_detail', sentence_id=sentence_id))

    try:
        # Pass only non-empty fields to the database function
        update_word_in_db(
            sentence_id=sentence_id,
            word=word,
            pos=pos,
            dep=dep,
            head=head,
            definitions=definitions if definitions else None,
            synonyms=synonyms if synonyms else None,
            antonyms=antonyms if antonyms else None
        )
        flash('Word details updated successfully.', 'success')
    except Exception as e:
        flash(f'An error occurred: {e}', 'danger')

    return redirect(url_for('sentence_detail', sentence_id=sentence_id))


@app.route('/sentence/<int:sentence_id>')
def sentence_detail(sentence_id):
    sentence_data = fetch_sentence_details(sentence_id)
    if sentence_data is None:
        return "Sentence not found", 404

    sentence= sentence_data['sentence']
    tone = sentence_data['tone']
    image_data = sentence_data['image']
    words = sentence_data['words']
    return render_template('sentence_detail.html',sentence_id=sentence_id, sentence=sentence,
                           tone=tone, image_data=image_data, words=words)

@app.route('/add_sentence', methods=['POST'])
def add_sentence():
    sentence_text = request.form.get('sentence')
    if not sentence_text:
        flash('Please enter a sentence.', 'danger')
        return redirect('/')

    try:
        # Call the function to save the sentence to the database
        save_sentences_to_db([sentence_text])
        flash('Sentence added successfully!', 'success')
    except Exception as e:
        flash(f'An error occurred: {e}', 'danger')

    return redirect('/')


@app.route('/user_manual')
def user_manual():
    return render_template('user_guide.html')


if __name__ == '__main__':
    app.run(debug=True)