import os
import spacy
import psycopg2
from collections import defaultdict

# Load SpaCy model
nlp = spacy.load("en_core_web_trf")

# Database connection
conn = psycopg2.connect(
    dbname='words_articles',
    user='postgres',
    password='1234',
    host='localhost',
    port='5432'
)
cursor = conn.cursor()

# Directory containing text files
directory = 'scrapper/articles'

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

        lines = text.split('\n')

        article_name = ""
        source_url = ""
        authors = ""
        publish_date = ""
        article_text = ""

        capture_text = False

        def clean_names_to_string(combined_names):
            split_names = combined_names.split(",")

            cleaned_names = set()

            for name in split_names:
                cleaned_name = name.strip()
                if not cleaned_name.startswith("More About"):
                    cleaned_names.add(cleaned_name)

            return ', '.join(cleaned_names)

        for line in lines:
            if capture_text:
                article_text += line + "\n"
            if line.startswith("Title:"):
                article_name = line.replace("Title:", "").strip()
            elif line.startswith("Authors:"):
                authors = line.replace("Authors:", "").strip()
            elif line.startswith("Published Date:"):
                publish_date = line.replace("Published Date:", "").strip()
            elif line.startswith("Source URL:"):
                source_url = line.replace("Source URL:", "").strip()
            elif line.strip() == "":
                capture_text = True

        authors = clean_names_to_string(authors)

        doc = nlp(article_text)

        cursor.execute("""
            INSERT INTO articles (name, source_url, authors, publish_date)
            VALUES (%s, %s, %s, %s) RETURNING id;
        """, (article_name, source_url, authors, publish_date))
        article_id = cursor.fetchone()[0]

        new_unique_words = defaultdict(lambda: (None, 0))
        word_counts = defaultdict(int)

        for token in doc:
            if token.is_alpha and not token.is_punct and not token.is_space:
                word_counts[token.text] += 1
                number = token.morph.get("Number")
                number_str = ', '.join(number) if number else 'N/A'

                if token.pos_ == 'VERB':
                    is_passive = any(child.dep_ in ["nsubj:pass", "auxpass", "agent"] for child in token.children)
                    voice_str = "Passive" if is_passive else "Active"
                    tense = token.morph.get("Tense")
                    tense_str = ', '.join(tense) if tense else 'N/A'

                    if any(child.text in ["will", "shall"] for child in token.children):
                        tense_str = "Future"

                    new_unique_words[token.text] = (
                        f"POS: {token.pos_}, LEMMA: {token.lemma_}, NUMBER: {number_str}, "
                        f"VOICE: {voice_str}, TENSE: {tense_str}",
                        word_counts[token.text],
                    )
                elif token.pos_ in ['NOUN', 'PRON']:
                    possessed_by_ = next((child.text for child in token.children if child.dep_ in ["poss", "nmod:poss"]), None)
                    possessor_of = token.head.text if token.dep_ in ["poss", "nmod:poss"] else None
                    case_str = f"Possessed by: {possessed_by_}" if possessed_by_ else f"Possesses: {possessor_of}" if possessor_of else "N/A"

                    new_unique_words[token.text] = (
                        f"POS: {token.pos_}, LEMMA: {token.lemma_}, NUMBER: {number_str}, CASE: {case_str}",
                        word_counts[token.text],
                    )
                else:
                    new_unique_words[token.text] = (
                        f"POS: {token.pos_}, LEMMA: {token.lemma_}, NUMBER: N/A",
                        word_counts[token.text],
                    )

        for word, (pos_info, count) in new_unique_words.items():
            cursor.execute("""
                INSERT INTO words (article_id, word, pos_info, count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (article_id, word) DO UPDATE SET count = words.count + EXCLUDED.count;
            """, (article_id, word, pos_info, count))


def process_file_and_save(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

        # Extract metadata using string manipulation
        lines = text.split('\n')

        # Initialize variables for metadata
        article_name = ""
        source_url = ""
        authors = ""
        publish_date = ""
        article_text = ""

        # Flag to indicate when to start capturing article text
        capture_text = False

        # Iterate through the lines to find metadata
        for line in lines:
            if capture_text:
                article_text += line + "\n"  # Append the article text
            if line.startswith("Title:"):
                article_name = line.replace("Title:", "").strip()
            elif line.startswith("Authors:"):
                authors = line.replace("Authors:", "").strip()
            elif line.startswith("Published Date:"):
                publish_date = line.replace("Published Date:", "").strip()
            elif line.startswith("Source URL:"):
                source_url = line.replace("Source URL:", "").strip()
            elif line.strip() == "":  # Blank line indicates end of metadata
                capture_text = True

            # Check for missing metadata
        if not article_name or not source_url or not authors or not publish_date:
            raise ValueError("Missing required metadata: Title, Authors, Published Date, or Source URL")

        doc = nlp(article_text)

        # Insert article metadata
        cursor.execute("""
            INSERT INTO articles (name, source_url, authors, publish_date)
            VALUES (%s, %s, %s, %s) RETURNING id;
        """, (article_name, source_url, authors, publish_date))
        article_id = cursor.fetchone()[0]

        new_unique_words = defaultdict(lambda: (None, 0))
        word_counts = defaultdict(int)

        for token in doc:
            if token.is_alpha and not token.is_punct and not token.is_space:
                word_counts[token.text] += 1
                number = token.morph.get("Number")
                number_str = ', '.join(number) if number else 'N/A'

                if token.pos_ == 'VERB':
                    is_passive = any(child.dep_ in ["nsubj:pass", "auxpass", "agent"] for child in token.children)
                    voice_str = "Passive" if is_passive else "Active"
                    tense = token.morph.get("Tense")
                    tense_str = ', '.join(tense) if tense else 'N/A'

                    if any(child.text in ["will", "shall"] for child in token.children):
                        tense_str = "Future"

                    new_unique_words[token.text] = (
                        f"POS: {token.pos_}, LEMMA: {token.lemma_}, NUMBER: {number_str}, "
                        f"VOICE: {voice_str}, TENSE: {tense_str}",
                        word_counts[token.text],
                    )
                elif token.pos_ in ['NOUN', 'PRON']:
                    possessed_by_ = next((child.text for child in token.children if child.dep_ in ["poss", "nmod:poss"]), None)
                    possessor_of = token.head.text if token.dep_ in ["poss", "nmod:poss"] else None
                    case_str = f"Possessed by: {possessed_by_}" if possessed_by_ else f"Possesses: {possessor_of}" if possessor_of else "N/A"

                    new_unique_words[token.text] = (
                        f"POS: {token.pos_}, LEMMA: {token.lemma_}, NUMBER: {number_str}, CASE: {case_str}",
                        word_counts[token.text],
                    )
                else:
                    new_unique_words[token.text] = (
                        f"POS: {token.pos_}, LEMMA: {token.lemma_}, NUMBER: N/A",
                        word_counts[token.text],
                    )

        # Insert words into the words table
        for word, (pos_info, count) in new_unique_words.items():
            cursor.execute("""
                INSERT INTO words (article_id, word, pos_info, count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (article_id, word) DO UPDATE SET count = words.count + EXCLUDED.count;
            """, (article_id, word, pos_info, count))

        conn.commit()
        cursor.close()
        conn.close()



if __name__ == "__main__":
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            process_file(os.path.join(directory, filename))
    conn.commit()
    cursor.close()
    conn.close()