# Project 4B Classifier

### Name

_Write your name here_

### Introspection

_Describe the challenges you faced and what you learned_

### Resources

_List the people and resources you used to complete the project_


### *DO NOT EDIT BELOW THIS LINE*
---

## Goal

The goals of this project are:

* creating custom functions
* using I/O to read files
* training a classifier!

## Description

In this project, you will write more functions for the IMDB movie review dataset and create classifier that predicts if a movie review is positive or negative. You will create a Naive Bayes Classifier to predict if the movie reviews are positive or negative. In this project, you will write fewer and slightly simpler functions. But you will have to call those functions more than once. This is a very simple classifier and implementing it form scratch is impressive. 


### `classifier.py`

Create a Python file called `classifier.py` that will contain all the following functions.


### `get_data`

Write a function called `get_data`. This function takes in two parameters - a list of movie reviews and a `sentiment`. The `sentiment` can either be positive or negative. In this context, we call `positive` and `negative` as labels of each review. This function returns a list of reviews that only have positive label or the negative label depending on the value of the parameter `sentiment`. Remember that in our movie reviews, the last word after each review is positive or negative indicating if the movie review has positive sentiment or negative sentiment. 

```
>>> reviews: list[str] = [
    "This was such an amazing MOVIE!!, positive",
    "This was a horrible movie. Not recommended at all., negative"
    "This movie is legendary. This is a classic movie., positive
]

>>> list_of_positive_reviews = get_data(reviews, "positive")
>>> print(list_of_positive_reviews)
[
    "This was such an amazing MOVIE!!",
    "I am such a big fan of this movie, it is amazing"
]
>>> list_of_negative_reviews = get_data(reviews, "negative")
>>> print(list_of_negative_reviews)
[
    "This was a horrible movie. Not recommended at all."
]
```

### `create_word_count_table`


Create a function that takes in a list of reviews of a specfic sentiment (positive or negative) and returns a dictionary where the key is a word and the value is the number times it occurs in all the reviews of a specfic sentiment (either positive or negative). You should call your `preprocess_text` function from the previous project to first preprocess each review from the list and then update the dictionary. 

```
... # continued from before
>>> print(list_of_positive_reviews)
[
    "This was such an amazing MOVIE!!",
    "I am such a big fan of this movie, it is amazing"
]
>>> positive_sentiment_word_count = create_word_count_table(list_of_positive_reviews)
>>> print(positive_sentiment_word_count)
{
    'movie': 2,
    'fan': 1,
    'amazing': 1,
    'such': 2,
    ...

}
... # Run this function for `list_of_negative_reviews` as well

```

Note that in the output, `MOVIE!!!` and `movie` are counted as two instances of the word `movie` that occur in the positive reviews. Remember to call the function on `list_of_negative_reviews` as well!


### `calculate_likelihood_for_each_word`

Create a function that again takes in two things - the dictionary from the previous function (`create_word_count_table`) and total words in the list of reviews for that sentiment. This function returns another dictionary called `likelihood`. In this dictionary the key is the word from the previous dictionary and the value is the probability of selecting that word from all the words in the reviews of that sentiment.

Here are the steps in English on how to do that:

```
# create likelihood dictionary

# unique_words = len(word_count_dictionary.keys())

for each_word in the word_count_dictionary:

    probability = (word_count_dictionary[word]  +  1) / total_words + unique_words
    likelihood[word] = probability

return likelihood
```


```
...
>>> print(positive_sentiment_word_count)
{
    'movie': 2,
    'fan': 1,
    'amazing': 1,
    'such': 2,
    ...

}
>>> len_positive_words = sum(positive_sentiment_word_count.values())
>>> prob_positive_words = calculate_likelihood_for_each_word(positive_sentiment_word_count, len_positive_words)
>>> print(prob_positive_words)
{
    'movie': 0.154, # your values could differ from this
    'fan': 0.076, # your values could differ from this
    'amazing': 0.076, # your values could differ from this
    'such': 0.154, # your values could differ from this
    ...
}

```

### `calculate_prior_probability`

This is the simplest functions to write. It takes in two parameters - list_of_reviews and `sentiment` similar to `get_data`. This function returns the ratio of number of reviews with the label `sentiment` to total reviews in the set. 


```
...
>>> print(reviews)
[
    "This was such an amazing MOVIE!!, positive",
    "This was a horrible movie. Not recommended at all., negative"
    "This movie is legendary. This is a classic movie., positive
]
>>> positive_review_prior_probability = calculate_prior_probability(reviews, "positive") 
>>> print(positive_review_prior_probability)
0.66667
>>> negative_review_prior_probability = calculate_prior_probability(reviews, "negative") 
>>> print(negative_review_prior_probability)
0.33333
```


### predict_label

This is the final function that predicts the label of the review based on its knowledge about the reviews dataset. This function takes in a lot of parameters. It takes in -

* `review`: str - This is the review that you want to predict the label for. 
* `prior_probability_positive`: float - Probability that the review is positive.
* `prior_probability_negative`: float - Probability that the review is negative.
* `likelihood_positive_words`: dict[str, float] - Dictionary containing the likelihood of words in positive reviews (i.e., output of `calculate_likelihood_for_each_word`).
* `likelihood_negative_words`: dict[str, float] - Dictionary containing the likelihood of words in negative reviews (i.e., output of `calculate_likelihood_for_each_word` but with negative reviews).


To predict the label calculate the probability for each label (i.e., `positive` and `negative`) and identify which probability is higher. Here is the formula to calculate the probability for each label.

$$
P(review | positive) = P(positive) * P(word_1 | positive) * P(word_2 | positive) \ldots * P(word_n | positive) \newline
P(review | negative) = P(negative) * P(word_1 | negative) * P(word_2 | negative) \ldots * P(word_n | negative)
$$

If the $P(review | positive)$ is greater than $P(review | negative)$ then return the label as `positive` otherwise return `negative`. 


### Final Code

After writing all the functions, you should call them in the following sequence at the end of your python file.

1. Load all the reviews from the `test_reviews.csv` or the `IMDB Dataset.csv`.
2. Create a list of all positive reviews by calling `get_data` function on the list of all reviews.
3. Create a list of all negative reviews by calling `get_data` function on the list of all reviews. 
4. Call `create_word_count_table` for both positive and negative review list.
5. Use the dictionaries from the previous steps to call `calculate_likelihood_for_each_word` for both positive and negative reviews.
6. Calculate the `prior_probability` for `positive` and `negative` reviews. 
7. Select a review randomly from the `IMDB Dataset.csv` and save it in a variable. Call `predict_label` on this variable
and pass in the necessary parameters that you computed in Steps 5 and 6. 


### `calculate_accuracy` (BONUS)

Write a function that calculates the accuracy of your classifier. For this function, you will first have to create two different sets of the list of movie reviews. The first set should, which we will call the training set, should contain 80% of all the reviews and the second set, which we will call test set, should contain 20% of all the reviews. Next, complete steps 1 - 6 for the training set. Then call the `predict_label` function on all the reviews from the test set and save their predictions in a list. To calculate the accuracy, you then need to compute the ratio of correct predictions to the total number of reviews in the test set i.e. (correct prediction / total reviews in test set).

A prediction is counted as correct, if its original label and its predicted label are same. For example, if a review is `negative` and the `predict_label` function predicts it as `negative` then this is counted as a correct prediction.


## Rubric

**1. Code Quality and Documentation** - **10 points**

- **Code Readability and Organization**: *5 points*
  - Meaningful variable and function names
- **Comments and Docstrings**: *5 points*
  - Clear explanations of functions and complex code sections
  - Proper use of docstrings for all functions

**2. Version Control Practice** - **10 points**

- **Commit Numbers and Sizes**: *5 points*
  - The commits are made at regular intervals and of coherent pieces 
- **Commit Messages**: *5 points*
  - Descriptive messages that are easy to follow along.


**3. `get_data` ** - **10 points**

**4. `create_word_count_table` ** - **10 points**

**5. `calculate_likelihood_for_each_word` ** - **10 points**

**6. `calculate_prior_probability` ** - **10 points**

**7. `predict_label` ** - **30 points**

**8. Final code - **40 points**


## Tips On How To Excel


* Start early!
* Ask for help when stuck. Remember the 30 minute rule? No? Look into the syllabus.
* Break down the problem into smaller tasks and try to implement them in Jupyter Notebook. Once implemented in the notebook successfully, transfer it into `.py` file.
* Run the `.py` file to make sure the new addition did not break any changes.
* Use DEBUGGER!!! To test and debug your code. Ensure that the variables are changing as they should be.
* After implementing each small task, commit changes.
* Review the notebooks from classes available on GitHub if you cannot remember syntax for anything.
* Run your code multiple times and vary the inputs to ensure it works as intended. 


## Feedback



