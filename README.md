# Naive Bayes Sentiment Classifier

This project builds a Naive Bayes classifier from scratch to predict the sentiment of IMDB movie reviews as **positive** or **negative**. It includes vocabulary creation, prior and likelihood computation, and numeric stability handling using log probabilities. Developed for **CS154**, an introductory Python course with NLP components.

---

## 📌 Features

- End-to-end implementation of a text classification pipeline:
  - Preprocess reviews and extract vocabulary
  - Build word count tables by sentiment
  - Compute prior and likelihood probabilities
  - Handle unseen words with fallback smoothing
  - Use log-probabilities to avoid numerical underflow
- Predict labels using the Naive Bayes rule
- Bonus: Accuracy evaluation on a train/test split

---

## 🧠 Reflections

- **Handling Unseen Words**: Initially assumed a probability of 1 for unknown words, but this led to overconfident predictions. Switched to `e^-6` (≈0.00247) for better performance.
- **Avoiding Underflow**: Learned about numerical instability caused by multiplying small probabilities. Used logarithmic addition to stabilize the classifier.
- This project solidified my understanding of probabilistic modeling and numerical precision in machine learning.

---

## 🛠️ How to Use

1. Clone the repository and ensure your dataset (e.g., `IMDB Dataset.csv`) is available.

2. Run the classifier:
   ```bash
   python classifier.py
   ```

3. The script performs:
   - Vocabulary building
   - Prior and likelihood computation
   - Sentiment prediction on a sample review

4. For the bonus task, the code also evaluates model accuracy over a test set split from the dataset.

---

## 📚 Resources

- [NLTK Documentation](https://www.nltk.org)
- [Stack Overflow on Naive Bayes & Numerical Underflow](https://stackoverflow.com)
- Functions reused from Project 4A: `preprocess_text`, `create_vocabulary`, etc.

---

## 🪪 License

This project is licensed under the MIT License.
