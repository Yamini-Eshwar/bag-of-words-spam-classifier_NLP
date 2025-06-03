### ❌ Why Not Label Encoding or One-Hot Encoding?

- **Label Encoding**: Converts categories to numbers (e.g., "spam" → 1, "ham" → 0).  
  ⛔ Not good for raw text because it falsely introduces **order/priority** between words.

- **One-Hot Encoding**: Converts each word into a long vector with 1 in the word’s position.  
  ⛔ Not scalable for large vocabularies. Too sparse and memory-heavy for emails with thousands of unique words.

### ✅ Why Use Bag of Words?

- It counts how often each word appears, without worrying about order.
- Works well with Naive Bayes, which assumes features are independent.
- Much more **compact**, **scalable**, and **effective** for text classification than OneHot or LabelEncoder.

# 📧 Spam Email Classifier using Bag of Words and Naive Bayes

This notebook is a simple demonstration of how to build a **spam classifier** using the **Bag of Words model** and a **Naive Bayes algorithm** with scikit-learn.  
It's ideal for beginners in NLP and machine learning.

---

## 📚 Concepts Used

### 🔹 Bag of Words (BoW)
- Converts raw text (emails) into a matrix of token counts.
- Each email becomes a vector of word frequencies.
- Words lose order/meaning, but we gain structured numeric data.

### 🔹 CountVectorizer
- Tokenizes the text and builds a vocabulary of known words.
- Returns the count of words in each email.
  
### 🔹 Naive Bayes Classifier
- Probabilistic algorithm that works well with word counts.
- Assumes word positions are independent (which works surprisingly well in text data).

---

## 🔧 How the Pipeline Works

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

clf = Pipeline([
    ('v', CountVectorizer()),        # Converts text to word counts
    ('nb', MultinomialNB())          # Trains a model on those word counts
])

clf.fit(X_train, y_train)


