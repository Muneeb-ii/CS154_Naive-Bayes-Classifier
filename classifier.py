from language_model import preprocess_text, create_vocabulary
from helper import get_file_contents


def get_data(reviews: list[str], sentiment) -> list[str]:
    i = 0
    reviews_without_sentiments: list[str] = []
    reviews_sentiments: list[str] = [each_review[-8:] for each_review in reviews]
    while i < len(reviews):
        if reviews_sentiments[i] == sentiment:
            reviews_without_sentiments.append(reviews[i][: len(reviews[i]) - 10])
            i += 1
        else:
            i += 1
    return reviews_without_sentiments


def create_word_count_table(
    list_of_sentiment_reviews: list[str], vocabulary: set[str]
) -> dict[str, int]:
    preprocessed_list_of_reviews: list[str] = [
        preprocess_text(each_review) for each_review in list_of_sentiment_reviews
    ]
    sentiment_word_counter: dict[str, int] = {}
    for each_word in vocabulary:
        for each_preprocessed_review in preprocessed_list_of_reviews:
            sentiment_word_counter.update(
                {
                    each_word: sentiment_word_counter.get(each_word, 0)
                    + each_preprocessed_review.count(each_word)
                }
            )
    return sentiment_word_counter


def calculate_likelihood_for_each_word(
    sentiment_word_count, len_sentiment_words
) -> dict[str:float]:
    unique_words: int = len(sentiment_word_count.keys())
    likelihood: dict[str:float] = {}
    for each_word in sentiment_word_count:
        probability: int = (sentiment_word_count.get(each_word) + 1) / (
            len_sentiment_words + unique_words
        )
        likelihood.update({each_word: round(probability, 3)})
    return likelihood


def calculate_prior_probability(reviews: list[str], sentiment: str) -> float:
    reviews_sentiments: list[str] = [each_review[-8:] for each_review in reviews]
    prior_probability: float = round(
        reviews_sentiments.count(sentiment) / len(reviews_sentiments), 5
    )
    return prior_probability
