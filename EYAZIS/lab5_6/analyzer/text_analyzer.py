import spacy
from nltk.corpus import wordnet as wn
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка моделей
try:
    nlp = spacy.load('en_core_web_lg')
    logger.info("spaCy model 'en_core_web_lg' loaded successfully")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}")
    raise

try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('wordnet', quiet=True)
    sia = SentimentIntensityAnalyzer()
    logger.info("NLTK resources downloaded and analyzer initialized")
except Exception as e:
    logger.error(f"Failed to initialize NLTK components: {str(e)}")
    raise


def analyze_text(text):
    """Анализирует текст и возвращает структурированные данные."""
    logger.info(f"Starting analysis of text: {text[:50]}...")

    if not text or not isinstance(text, str):
        logger.warning("Empty or non-string text provided for analysis")
        return None

    try:
        doc = nlp(text.lower().strip())
        logger.debug(f"Processed text with spaCy: {len(doc)} tokens")

        meaningful_tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]

        analysis = {
            'sentiment': sia.polarity_scores(text),
            'keywords': meaningful_tokens,
            'entities': " ".join([ent.text for ent in doc.ents]),
        }

        logger.info(f"Found {len(analysis['entities'])} entities and {len(analysis['keywords'])} keywords")

        # Detailed log of the analysis results
        logger.info("Analysis results summary:")
        logger.info(f"Sentiment scores - Positive: {analysis['sentiment']['pos']:.2f}, "
                   f"Negative: {analysis['sentiment']['neg']:.2f}, "
                   f"Neutral: {analysis['sentiment']['neu']:.2f}, "
                   f"Compound: {analysis['sentiment']['compound']:.2f}")
        logger.info(f"Top 5 keywords: {analysis['keywords'][:5]}")
        logger.info(f"Entities found: {analysis['entities']}")

        return analysis

    except Exception as e:
        logger.error(f"Error during text analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
    The company designs, develops, and sells consumer electronics, computer software, and online services. 
    It's considered one of the most innovative companies in the world, though some critics argue about its environmental policies.
    """

    print("Analyzing sample text...")
    result = analyze_text(sample_text)

    print("\nAnalysis Results:")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Keywords: {result['keywords']}")
    print(f"Entities: {result['entities']}")