import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

include_logger = logging.getLogger('include')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def case_normalization(text):
    include_logger.debug("case_normalization called")
    return text.lower()


def remove_stopwords(text):
    include_logger.debug("remove_stopwords called")
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]

    return ' '.join(filtered_words)


def stem_text(text):
    include_logger.debug("stem_text called")
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]

    return ' '.join(stemmed_words)


def preprocess_text(text):
    include_logger.debug("preprocess_text called")
    text = case_normalization(text)
    text = remove_stopwords(text)
    text = stem_text(text)

    return text
