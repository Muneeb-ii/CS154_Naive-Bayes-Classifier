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
