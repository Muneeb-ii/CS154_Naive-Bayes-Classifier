from language_model import preprocess_text, create_vocabulary


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
            if each_word in sentiment_word_counter:
                sentiment_word_counter.update(
                    {
                        each_word: sentiment_word_counter.get(each_word)
                        + each_preprocessed_review.count(each_word)
                    }
                )
            else:
                sentiment_word_counter.update(
                    {each_word: each_preprocessed_review.count(each_word)}
                )
    return sentiment_word_counter
