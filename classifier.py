def get_data(reviews: list[str], sentiment) -> list[str]:
    i = 0
    required_list: list[str] = []
    reviews_sentiments: list[str] = [
        each_review.split(",")[1].strip() for each_review in reviews
    ]
    while i < len(reviews):
        if reviews_sentiments[i] == sentiment:
            required_list.append(reviews[i].split(",")[0].strip())
            i += 1
        else:
            i += 1
    return required_list
