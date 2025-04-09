import psycopg2
import spacy
from psycopg2 import IntegrityError
from spacy import displacy
import imgkit
import os
import base64
from nltk.corpus import wordnet as wn
from striprtf.striprtf import rtf_to_text
from textblob import TextBlob

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


def extract_sentences_from_doc(file_path):
    pass


def extract_sentences_from_rtf(file_path):
    with open(file_path, 'r') as rtf_file:
        rtf_content = rtf_file.read()
        plain_text = rtf_to_text(rtf_content)
        sentences = [sentence.strip() for sentence in plain_text.split('.')]
        return sentences

def get_database_connection():
    return psycopg2.connect(
        dbname='semantic_analysis',
        user='postgres',
        password='1234',
        host='localhost',
        port='5432'
    )


def analyze_sentence(sentence_text):
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Strip and process the sentence
    sentence_text_stripped = sentence_text.strip()
    doc = nlp(sentence_text_stripped)

    # Render dependency parse tree
    html = displacy.render(doc, style='dep', jupyter=False)

    with open("temp_tree.html", "w") as f:
        f.write(html)

    img_data = imgkit.from_file('temp_tree.html', False, config=imgkit.config(
        wkhtmltoimage='C:\\programmes\\wkhtmltopdf\\bin\\wkhtmltoimage.exe'))

    # Clean up temporary file
    os.remove("temp_tree.html")

    # Analyze sentiment using TextBlob
    blob = TextBlob(sentence_text)
    sentiment = blob.sentiment
    if sentiment.polarity > 0:
        tone = "Positive"
    elif sentiment.polarity < 0:
        tone = "Negative"
    else:
        tone = "Neutral"

    # Prepare tokens data with full names
    tokens_data = []
    for token in doc:
        if not token.is_punct or not token.is_space:
            word = token.text
            synonyms = wn.synsets(word)

            # Use a set to collect unique synonyms
            synonym_words = {syn.lemmas()[0].name() for syn in synonyms}

            # Get definitions
            definitions = [syn.definition() for syn in synonyms]
            definitions = definitions[:3]
            # Get antonyms
            antonyms = set()  # Use a set for antonyms too
            for syn in synonyms:
                for lemma in syn.lemmas():
                    if lemma.antonyms():
                        antonyms.add(lemma.antonyms()[0].name())


            # Create token data entry
            token_info = {
                'word': word,
                'pos': pos_full_names.get(token.pos_, token.pos_),
                'dep': dep_full_names.get(token.dep_, token.dep_),
                'head': token.head.text,
                'definitions': definitions,
                'synonyms': list(synonym_words),
                'antonyms': list(antonyms)
            }
            tokens_data.append(token_info)

    return img_data, tokens_data, tone

def save_sentences_to_db(sentences):
    """Saves a list of sentences and their analyses to the database."""
    conn = get_database_connection()  # Ensure you have a valid connection function
    cur = conn.cursor()

    for sentence_text in sentences:
        print(f"Processing sentence: '{sentence_text}'")

        try:
            # Analyze the sentence
            img_data, tokens_data, tone = analyze_sentence(sentence_text)

            # Insert the sentence into the database
            cur.execute(
                "INSERT INTO sentences (sentence, tone, image) VALUES (%s, %s, %s) RETURNING id;",
                (sentence_text, tone, psycopg2.Binary(img_data))
            )
            sentence_id = cur.fetchone()[0]
            print(f"Sentence inserted with ID: {sentence_id}")

            # Insert each token's information into the database
            for token_data in tokens_data:
                cur.execute(
                    """
                    INSERT INTO words (
                        sentence_id, word, pos, dep, head, definitions, synonyms, antonyms
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        sentence_id,
                        token_data['word'],
                        token_data['pos'],
                        token_data['dep'],
                        token_data['head'],
                        token_data['definitions'],
                        token_data['synonyms'],
                        token_data['antonyms'],
                    )
                )

        except psycopg2.IntegrityError:
            # Handle duplicate sentence case
            print(f"Sentence '{sentence_text}' already exists in the database.")
            conn.rollback()
        except Exception as e:
            # Handle other exceptions
            print(f"Error processing sentence '{sentence_text}': {e}")
            conn.rollback()
        else:
            # Commit the transaction for the current sentence
            conn.commit()

    print("All sentences have been successfully processed.")

    # Close the cursor and connection
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


def fetch_sentence_details(sentence_id):
    """Fetch the details of a particular sentence including its image, tone, and associated words."""
    conn = get_database_connection()
    cur = conn.cursor()
    sentence_details = None

    try:
        # Fetch the sentence, tone, and its associated image
        cur.execute("SELECT sentence, tone, image FROM sentences WHERE id = %s;", (sentence_id,))
        sentence_data = cur.fetchone()

        if sentence_data:
            sentence_text, tone, image_data = sentence_data

            # Encode the image data to base64
            encoded_image = base64.b64encode(image_data).decode('utf-8') if image_data else None

            # Fetch all words associated with this sentence
            cur.execute(
                """
                SELECT word, pos, dep, head, definitions, synonyms, antonyms 
                FROM words 
                WHERE sentence_id = %s;
                """,
                (sentence_id,)
            )
            words_data = cur.fetchall()


            words = []
            for word, pos, dep, head, definitions, synonyms, antonyms in words_data:
                words.append({
                    'word': word,
                    'pos': pos,
                    'dep': dep,
                    'head': head,
                    'definitions': '; '.join(definitions),  # Join definitions with '; '
                    'synonyms': '; '.join(synonyms),  # Join synonyms with '; '
                    'antonyms': '; '.join(antonyms)
                })

            # Prepare the sentence details
            sentence_details = {
                'sentence': sentence_text,
                'tone': tone,
                'image': encoded_image,
                'words': words
            }
    except Exception as e:
        print(f"An error occurred while fetching sentence details: {e}")
    finally:
        cur.close()
        conn.close()

    return sentence_details
def update_word_in_db(sentence_id, word, pos=None, dep=None, head=None, definitions=None, synonyms=None, antonyms=None):
    try:
        # Establish a connection to the database
        conn = get_database_connection()
        cur = conn.cursor()

        # Start constructing the SQL query dynamically
        query = "UPDATE words SET "
        params = []

        # Add fields to update only if they are provided
        if pos:
            query += "pos = %s, "
            params.append(pos)
        if dep:
            query += "dep = %s, "
            params.append(dep)
        if head:
            query += "head = %s, "
            params.append(head)
        if definitions !=['']:
            query += "definitions = %s, "
            params.append(definitions)  # Array field
        if synonyms !=['']:
            query += "synonyms = %s, "
            params.append(synonyms)    # Array field
        if antonyms !=['']:
            query += "antonyms = %s, "
            params.append(antonyms)    # Array field

        # Remove the trailing comma and space
        query = query.rstrip(", ")

        # Add WHERE clause to target the correct record
        query += " WHERE sentence_id = %s AND word = %s"
        params.extend([sentence_id, word])

        # Execute the query
        cur.execute(query, params)

        # Commit changes and close the connection
        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        raise Exception(f"Database Error: {e}")

if __name__ == "__main__":
    save_sentences_to_db(["i do not want to get ice cream with your dad"])