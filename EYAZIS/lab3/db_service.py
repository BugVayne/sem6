import psycopg2
import spacy
from psycopg2 import IntegrityError
from spacy import displacy
import imgkit
import os
from striprtf.striprtf import rtf_to_text
import base64


def update_word_in_db(sentence_id, word, pos, dep, head):
    """Update word information in the database."""
    conn = get_database_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE words SET pos = %s, dep = %s, head = %s WHERE sentence_id = %s AND word = %s;",
            (pos, dep, head, sentence_id, word)
        )
        conn.commit()
    except Exception as e:
        print(f"An error occurred while updating the word: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def extract_sentences_from_rtf(file_path):
    with open(file_path, 'r') as rtf_file:
        rtf_content = rtf_file.read()
        plain_text = rtf_to_text(rtf_content)
        # Split sentences by period and strip whitespace from each sentence
        sentences = [sentence.strip() for sentence in plain_text.split('. ')]
        return sentences

def get_database_connection():
    """Establish database connection and return the connection object."""
    return psycopg2.connect(
        dbname='structural_analysis',
        user='postgres',
        password='1234',
        host='localhost',
        port='5432'
    )

def analyze_sentence(sentence_text):
    """Analyzes a single sentence and returns its tokens and syntactic tree image data."""
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    sentence_text_stripped = sentence_text.strip()
    doc = nlp(sentence_text_stripped)

    html = displacy.render(doc, style='dep', jupyter=False)

    with open("temp_tree.html", "w") as f:
        f.write(html)

    img_data = imgkit.from_file('temp_tree.html', False, config=imgkit.config(
        wkhtmltoimage='C:\\programmes\\wkhtmltopdf\\bin\\wkhtmltoimage.exe'))

    os.remove("temp_tree.html")

    pos_full_names = {
        "ADJ": "Adjective",
        "ADP": "Adposition",
        "ADV": "Adverb",
        "AUX": "Auxiliary",
        "CCONJ": "Coordinating conjunction",
        "DET": "Determiner",
        "INTJ": "Interjection",
        "NOUN": "Noun",
        "NUM": "Numeral",
        "PART": "Particle",
        "PRON": "Pronoun",
        "PROPN": "Proper noun",
        "PUNCT": "Punctuation",
        "SCONJ": "Subordinating conjunction",
        "SYM": "Symbol",
        "VERB": "Verb",
        "X": "Other",
    }

    dep_full_names = {
        "nmod": "Nominal modifier",
        "oprd": "Object of a preposition",
        "punct": "Punctuation",
        "acl": "Clausal modifier of noun",
        "advcl": "Adverbial clause modifier",
        "advmod": "Adverbial modifier",
        "amod": "Adjectival modifier",
        "appos": "Appositional modifier",
        "attr": "Attribute",
        "aux": "Auxiliary",
        "auxpass": "Passive auxiliary",
        "cc": "Coordinating conjunction",
        "ccomp": "Clausal complement",
        "compound": "Compound",
        "conj": "Conjunct",
        "det": "Determiner",
        "dobj": "Direct object",
        "expl": "Expletive",
        "feel": "Feeler",
        "iobj": "Indirect object",
        "mark": "Marker",
        "nsubj": "Nominal subject",
        "nsubjpass": "Nominal subject (passive)",
        "pobj": "Object of preposition",
        "poss": "Possessor",
        "prep": "Preposition",
        "prune": "Pruned",
        "root": "Root",
        "tmod": "Temporal modifier",
        "vocative": "Vocative",
        "xcomp": "Open clausal complement",
    }

    # Prepare tokens data with full names
    tokens_data = [
        (
            token.text,
            pos_full_names.get(token.pos_, token.pos_),  # Get full POS name, default to short name
            dep_full_names.get(token.dep_, token.dep_),  # Get full dep name, default to short name
            token.head.text
        )
        for token in doc
    ]

    return img_data, tokens_data


def save_sentences_to_db(sentences):
    """Saves a list of sentences and their analyses to the database."""
    conn = get_database_connection()
    cur = conn.cursor()

    for sentence_text in sentences:
        print(f"Processing sentence: '{sentence_text}'")

        img_data, tokens_data = analyze_sentence(sentence_text)

        try:
            # Try to insert the sentence into the database
            cur.execute("INSERT INTO sentences (sentence) VALUES (%s) RETURNING id;", (sentence_text,))
            sentence_id = cur.fetchone()[0]
            print(f"Sentence inserted with ID: {sentence_id}")

            # Update the database with the generated image
            cur.execute("UPDATE sentences SET image = %s WHERE id = %s;", (psycopg2.Binary(img_data), sentence_id))

            # Insert each token's information into the database
            for token_data in tokens_data:
                cur.execute(
                    "INSERT INTO words (sentence_id, word, pos, dep, head) VALUES (%s, %s, %s, %s, %s);",
                    (sentence_id, *token_data)
                )
        except IntegrityError:
            # Handle duplicate sentence case
            print(f"Sentence '{sentence_text}' already exists in the database.")
            conn.rollback()

    conn.commit()
    print("All sentences have been successfully processed.")

    cur.close()
    conn.close()


def fetch_sentences():
    """Fetch sentences along with their IDs from the database."""
    sentences = []
    try:
        # Connect to the database
        conn = get_database_connection()
        cur = conn.cursor()

        # Fetch sentences and their IDs
        cur.execute("SELECT id, sentence FROM sentences;")
        sentences = [{'id': row[0], 'text': row[1]} for row in cur.fetchall()]

        # Close cursor and connection
        cur.close()
        conn.close()
    except Exception as e:
        print(f"An error occurred while fetching sentences: {e}")

    return sentences

import base64

def fetch_sentence_details(sentence_id):
    """Fetch the details of a particular sentence including its image and words."""
    conn = get_database_connection()
    cur = conn.cursor()
    sentence_data = None
    try:
        # Fetch the sentence and its associated image
        cur.execute("SELECT sentence, image FROM sentences WHERE id = %s;", (sentence_id,))
        sentence_data = cur.fetchone()

        if sentence_data:
            sentence_text, image_data = sentence_data

            # Encode the image data to base64
            encoded_image = base64.b64encode(image_data).decode('utf-8') if image_data else None

            # Fetch all words associated with this sentence
            cur.execute("SELECT word, pos, dep, head FROM words WHERE sentence_id = %s;", (sentence_id,))
            words_data = cur.fetchall()
            return sentence_text, encoded_image, words_data
    finally:
        cur.close()
        conn.close()
    return None