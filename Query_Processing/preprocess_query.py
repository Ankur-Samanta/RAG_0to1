import string
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # Download the Open Multilingual WordNet
from nltk.corpus import wordnet
from spellchecker import SpellChecker

def normalize_query(query):
    """
    Normalizes a query by converting it to lowercase and removing punctuation.

    Parameters:
    - query (str): The query string to be normalized.

    Returns:
    - str: The normalized query.
    """
    # Convert query to lowercase.
    query = query.lower()
    # Remove punctuation from the query.
    query = query.translate(str.maketrans('', '', string.punctuation))
    return query

def expand_synonyms(word):
    """
    Finds and returns synonyms of a given word using NLTK's WordNet.

    Parameters:
    - word (str): The word for which synonyms are to be found.

    Returns:
    - list: A list of synonyms for the given word.
    """
    synonyms = set()
    # Iterate through all synsets of the word to gather synonyms.
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def correct_spelling(text):
    """
    Corrects the spelling of words in a given text using a spell checker.

    Parameters:
    - text (str): The text whose spelling needs to be corrected.

    Returns:
    - str: The text with corrected spelling.
    """
    spell = SpellChecker()
    # Split the text into individual words.
    words = text.split()
    # Correct the spelling of each word.
    corrected_words = [spell.correction(word) for word in words]
    # Re-join the corrected words into a single string.
    return ' '.join(corrected_words)

def preprocess_query(query):
    """
    Preprocesses a query by normalizing, spelling correction, and optionally expanding synonyms.

    Parameters:
    - query (str): The query to be preprocessed.

    Returns:
    - str: The preprocessed query.
    """
    # Normalize the query to lowercase and remove punctuation.
    normalized_query = normalize_query(query)
    # Correct spelling mistakes in the normalized query.
    corrected_query = correct_spelling(normalized_query)
    
    # Optionally, expand synonyms for key terms. This step is context-dependent and might not be applied universally.
    # Example of how synonyms could be expanded for each word, commented out for optional use.
    # words = corrected_query.split()
    # expanded_query = " ".join([expand_synonyms(word)[0] if expand_synonyms(word) else word for word in words])
    
    # In this implementation, the expanded_query step is not actively applied.
    expanded_query = corrected_query
    
    return expanded_query