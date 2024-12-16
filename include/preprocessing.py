import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def case_normalization(text):
    # logger.info("Normalizing case of the text...")
    return text.lower()


def remove_stopwords(text):
    # logger.info("Removing stopwords from the text...")
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


def stem_text(text):
    # logger.info("Stemming the text...")
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)


def preprocess_text(text):
    # logger.info("Starting text preprocessing...")
    text = case_normalization(text)
    # logger.info("Case normalization completed.")
    text = remove_stopwords(text)
    # logger.info("Stopword removal completed.")
    text = stem_text(text)
    # logger.info("Preprocessing completed.")
    return text
