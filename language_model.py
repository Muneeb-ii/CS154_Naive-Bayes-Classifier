from helper import get_file_contents
from nltk.stem import WordNetLemmatizer
import math

wnl = WordNetLemmatizer()

list_of_stop_words: str = get_file_contents("list_of_stop_words.txt")
characters_to_remove: list[str]= ["!", ",", ".", "?", ":", ";", "-", "_"]
translation_table = str.maketrans("", "", "".join(characters_to_remove))


def preprocess_text(text: str) -> list[str]:
    """
    Preprocesses a given text by converting it to lowercase, removing punctuation,
    tokenizing the text, removing stop words, and applying lemmatization.

    Args:
        text (str): The text to preprocess.

    Returns:
        list[str]: A list of preprocessed words.
    """
    text_lower: str = text.lower()
    text_no_punctuation: str = text_lower.translate(translation_table)
    text_tokenized: list[str] = text_no_punctuation.split(" ")
    text_no_stop_words: list[str] = [
        f for f in text_tokenized if f not in list_of_stop_words
    ]
    text_stemmed: list[str] = [wnl.lemmatize(f, pos="v") for f in text_no_stop_words]
    return text_stemmed


def get_unique_words(text: list[str]) -> set[str]:
    """
    Extracts a set of unique words from a preprocessed text.

    Args:
        text (list[str]): A list of preprocessed words.

    Returns:
        set[str]: A set of unique words in the text.
    """
    text_unique: set[str] = set(text)
    return text_unique


def create_vocabulary(reviews: list[str]) -> set[str]:
    """
    Creates a vocabulary set from a list of reviews by preprocessing each review
    and extracting the unique words.

    Args:
        reviews (list[str]): A list of reviews.

    Returns:
        set[str]: A set of unique words found in all the reviews.
    """
    vocabulary: list[str] = []
    for each_review in reviews:
        vocabulary.extend(get_unique_words(preprocess_text(each_review)))
    return set(vocabulary)


def calculate_term_frequency_for_each_review(
    preprocessed_review: list[str],
) -> dict[str, int]:
    """
    Calculates the term frequency for a single review.

    Args:
        preprocessed_review (list[str]): A list of preprocessed words in the review.

    Returns:
        dict[str, int]: A dictionary where the keys are words and the values are their frequencies in the review.
    """
    term_frequency: dict[str, int] = {}
    for each_word in preprocessed_review:
        term_frequency.update({each_word: preprocessed_review.count(each_word)})
    return term_frequency


def calculate_term_frequency_for_corpus(
    preprocessed_reviews: list[list[str]],
) -> list[dict[str, int]]:
    """
    Calculates the term frequency for a corpus of reviews.

    Args:
        preprocessed_reviews (list[list[str]]): A list of preprocessed reviews, where each review is a list of words.

    Returns:
        list[dict[str, int]]: A list of dictionaries, each containing the term frequencies for a review.
    """
    tf_corpus: list[dict[str, int]] = []
    for each_review in preprocessed_reviews:
        term_frequency: dict[str, int] = {}
        for each_word in each_review:
            term_frequency.update({each_word: each_review.count(each_word)})
        tf_corpus.append(term_frequency)
    return tf_corpus


def calculate_tf_idf(term_frequencies: list[dict], vocabulary: set[str]) -> list[dict]:
    """
    Calculates the TF-IDF score for each word in each review.

    Args:
        term_frequencies (list[dict]): A list of dictionaries containing the term frequencies for each review.
        vocabulary (set[str]): A set of unique words from the entire corpus.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains the TF-IDF scores for each word in a review.
    """
    tf_idf_corpus: list[dict] = []
    for review_term_frequencies in term_frequencies:
        doc_tf_idf: dict = {}
        for each_word in review_term_frequencies.keys():
            dft: int = 0
            TF: int = review_term_frequencies.get(each_word)
            i: int = 0
            while i < len(term_frequencies):
                if each_word in term_frequencies[i]:
                    dft = dft + term_frequencies[i].get(each_word)
                    i += 1
                else:
                    i += 1
            doc_tf_idf.update(
                {each_word: round(TF * math.log(len(term_frequencies) / dft), 4)}
            )
        tf_idf_corpus.append(doc_tf_idf)
    return tf_idf_corpus
