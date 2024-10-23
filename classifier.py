from language_model import preprocess_text, create_vocabulary
from helper import get_file_contents


def get_data(reviews: list[str], sentiment) -> list[str]:
    i = 0
    reviews_without_sentiments: list[str] = []
    reviews_sentiments: list[str] = [each_review[-8:] for each_review in reviews]
    while i < len(reviews):
        if reviews_sentiments[i] == sentiment:
            reviews_without_sentiments.append(reviews[i][1 : len(reviews[i]) - 10])
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


def predict_label(
    review: str,
    prior_probability_positive: float,
    prior_probability_negative: float,
    likelihood_positive_words: dict[str, float],
    likelihood_negative_words: dict[str, float],
) -> str:
    preprocessed_review: str = preprocess_text(review)
    prob_for_positive: float = prior_probability_positive
    prob_for_negative: float = prior_probability_negative
    for each_word in preprocessed_review:
        prob_for_positive = prob_for_positive * likelihood_positive_words.get(
            each_word, 1
        )
        prob_for_negative = prob_for_negative * likelihood_negative_words.get(
            each_word, 1
        )
    if prob_for_positive > prob_for_negative:
        return "Positive"
    else:
        return "Negative"


reviews: list[str] = get_file_contents("test_reviews.csv").splitlines()[1:]

list_of_positive_reviews = get_data(reviews, "positive")

list_of_negative_reviews = get_data(reviews, "negative")

vocabulary: set[str] = create_vocabulary(list_of_positive_reviews)
vocabulary_negative: set[str] = create_vocabulary(list_of_negative_reviews)
vocabulary.update(vocabulary_negative)

positive_sentiment_word_count = create_word_count_table(
    list_of_positive_reviews, vocabulary
)

negative_sentiment_word_count = create_word_count_table(
    list_of_negative_reviews, vocabulary
)

len_positive_words = sum(positive_sentiment_word_count.values())
prob_positive_words = calculate_likelihood_for_each_word(
    positive_sentiment_word_count, len_positive_words
)

len_negative_words = sum(negative_sentiment_word_count.values())
prob_negative_words = calculate_likelihood_for_each_word(
    negative_sentiment_word_count, len_negative_words
)

positive_review_prior_probability = calculate_prior_probability(reviews, "positive")
print(positive_review_prior_probability)
negative_review_prior_probability = calculate_prior_probability(reviews, "negative")
print(negative_review_prior_probability)

#Testing the predict_label function
test_review_1_IMBD: str = "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring. It just never gets old, despite my having seen it some 15 or more times in the last 25 years. Paul Lukas' performance brings tears to my eyes, and Bette Davis, in one of her very few truly sympathetic roles, is a delight. The kids are, as grandma says, more like ""dressed-up midgets"" than children, but that only makes them more fun to watch. And the mother's slow awakening to what's happening in the world and under her own roof is believable and startling. If I had a dozen thumbs, they'd all be ""up"" for this movie." #Positive 
test_review_2_IMBD: str = "The cast played Shakespeare.<br /><br />Shakespeare lost.<br /><br />I appreciate that this is trying to bring Shakespeare to the masses, but why ruin something so good.<br /><br />Is it because 'The Scottish Play' is my favorite Shakespeare? I do not know. What I do know is that a certain Rev Bowdler (hence bowdlerization) tried to do something similar in the Victorian era.<br /><br />In other words, you cannot improve perfection.<br /><br />I have no more to write but as I have to write at least ten lines of text (and English composition was never my forte I will just have to keep going and say that this movie, as the saying goes, just does not cut it." #Negative
print(predict_label(test_review_1_IMBD, positive_review_prior_probability, negative_review_prior_probability, prob_positive_words, prob_negative_words ))
print(predict_label(test_review_2_IMBD, positive_review_prior_probability, negative_review_prior_probability, prob_positive_words, prob_negative_words ))