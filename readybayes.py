import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


# Function to load text files from the given directory (IMDB structure: "pos" and "neg" subfolders)
def load_imdb_data(data_dir):
    texts, labels = [], []
    for label in ["pos", "neg"]:
        folder_path = os.path.join(data_dir, label)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                texts.append(file.read())
                labels.append(1 if label == "pos" else 0)
    return texts, np.array(labels)

# Get user input for training and testing folder paths
train_folder = input("Enter the path to the training data folder: ")
test_folder = input("Enter the path to the testing data folder: ")

# Get user input for n, m, k
n = int(input("Enter the number of most frequent words to ignore (n): "))
k = int(input("Enter the number of least frequent words to ignore (k): "))
m = int(input("Enter the number of words to keep based on Information Gain (m): "))

# Load data
X_train_full, y_train_full = load_imdb_data(train_folder)
X_test, y_test = load_imdb_data(test_folder)

# Split full training set into training (80%) and development (20%) sets
X_train, X_dev, y_train, y_dev = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Convert text data to binary features (Bag of Words) with CountVectorizer
vectorizer = CountVectorizer(binary=True, stop_words='english', max_features=4000)
X_train_matrix = vectorizer.fit_transform(X_train)

# Remove the n most frequent words and k least frequent words
word_freq = np.array(X_train_matrix.sum(axis=0)).flatten()
vocab = np.array(vectorizer.get_feature_names_out())

# Ignore the n most frequent words
most_frequent_indices = np.argsort(word_freq)[-n:]
filtered_vocab = np.delete(vocab, most_frequent_indices)
word_freq_filtered = np.delete(word_freq, most_frequent_indices)

# Ignore the k least frequent words
least_frequent_indices = np.argsort(word_freq_filtered)[:k]
filtered_vocab = np.delete(filtered_vocab, least_frequent_indices)
word_freq_filtered = np.delete(word_freq_filtered, least_frequent_indices)

# Keep the m words with the highest Information Gain (εδώ χρησιμοποιούμε απλά τη συχνότητα ως proxy)
top_m_indices = np.argsort(word_freq_filtered)[-m:]
final_vocab = filtered_vocab[top_m_indices]

# New CountVectorizer with the final vocabulary
vectorizer = CountVectorizer(binary=True, vocabulary=final_vocab)
X_train_binary = vectorizer.fit_transform(X_train)
X_dev_binary = vectorizer.transform(X_dev)

# Learning curves: track precision, recall, F1 for negatives
train_sizes = np.linspace(0.1, 1.0, 10)
train_precisions, train_recalls, train_f1s = [], [], []
dev_precisions, dev_recalls, dev_f1s = [], [], []

def compute_metrics(y_true, y_pred):
    #Consider negative class as 0
    ## True Negatives (TN): Correctly predicted negative samples
    tp = np.sum((y_pred == 0) & (y_true == 0))
    # False Positives (FP): Incorrectly predicted positives
    fp = np.sum((y_pred == 0) & (y_true == 1))
    # False Negatives (FN): Incorrectly predicted negatives
    fn = np.sum((y_pred == 1) & (y_true == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# Calculates precision, recall, and F1-score for the negative class (label 0) in a binary classification problem
for size in train_sizes:
    subset_size = int(size * len(y_train))
    X_train_subset = X_train_binary[:subset_size]
    y_train_subset = y_train[:subset_size]

    # Keep only the negative training samples
    neg_indices_train = np.where(y_train_subset == 0)[0]
    X_train_subset_neg = X_train_subset[neg_indices_train]
    y_train_subset_neg = y_train_subset[neg_indices_train]

    neg_indices_dev = np.where(y_dev == 0)[0]
    X_dev_neg = X_dev_binary[neg_indices_dev]
    y_dev_neg = y_dev[neg_indices_dev]

    # Train BernoulliNB on ολόκληρο το subset (και όχι μόνο τα αρνητικά)
    nb = BernoulliNB()
    nb.fit(X_train_subset, y_train_subset)

    # Predictions on negative samples
    y_train_pred_neg = nb.predict(X_train_subset_neg)
    y_dev_pred_neg = nb.predict(X_dev_neg)

    # Compute precision, recall, and F1 for negatives
    train_precision, train_recall, train_f1 = compute_metrics(y_train_subset_neg, y_train_pred_neg)
    dev_precision, dev_recall, dev_f1 = compute_metrics(y_dev_neg, y_dev_pred_neg)

    # Store results
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1s.append(train_f1)
    dev_precisions.append(dev_precision)
    dev_recalls.append(dev_recall)
    dev_f1s.append(dev_f1)

# Plot learning curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_sizes * 100, train_precisions, label="Train Precision", marker='o', color='blue')
axes[0].plot(train_sizes * 100, dev_precisions, label="Dev Precision", marker='o', linestyle="dashed", color='blue')
axes[0].set_xlabel("Training Data Percentage")
axes[0].set_ylabel("Precision")
axes[0].set_title("Precision Learning Curve")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(train_sizes * 100, train_recalls, label="Train Recall", marker='s', color='red')
axes[1].plot(train_sizes * 100, dev_recalls, label="Dev Recall", marker='s', linestyle="dashed", color='red')
axes[1].set_xlabel("Training Data Percentage")
axes[1].set_ylabel("Recall")
axes[1].set_title("Recall Learning Curve")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(train_sizes * 100, train_f1s, label="Train F1", marker='^', color='green')
axes[2].plot(train_sizes * 100, dev_f1s, label="Dev F1", marker='^', linestyle="dashed", color='green')
axes[2].set_xlabel("Training Data Percentage")
axes[2].set_ylabel("F1 Score")
axes[2].set_title("F1 Score Learning Curve")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Final evaluation on the Test Set
# Transform the test data using the final vectorizer
X_test_binary = vectorizer.transform(X_test)

# Train the final model using the entire training set (X_train_binary, y_train)
nb_final = BernoulliNB()
nb_final.fit(X_train_binary, y_train)

# Make predictions on the test set
y_test_pred = nb_final.predict(X_test_binary)

# Print Classification Report
report = classification_report(y_test, y_test_pred, target_names=["Negative", "Positive"])
print("Test Set Classification Report:\n")
print(report)