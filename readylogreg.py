import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


# Function to compute precision, recall, and F1-score for each class
def compute_metrics(y_true, y_pred):
    # For Negative class (label 0)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 1) & (y_pred == 0))
    fn = np.sum((y_true == 0) & (y_pred == 1))
    precision_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0

    # For Positive class (label 1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp_pos = np.sum((y_true == 0) & (y_pred == 1))
    fn_pos = np.sum((y_true == 1) & (y_pred == 0))
    precision_pos = tp / (tp + fp_pos) if (tp + fp_pos) > 0 else 0
    recall_pos = tp / (tp + fn_pos) if (tp + fn_pos) > 0 else 0
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

    # Counts the number of negative class samples (label 0)
    support_neg = np.sum(y_true == 0)
    # Counts the number of positive class samples (label 1)
    support_pos = np.sum(y_true == 1)
    
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    macro_precision = (precision_neg + precision_pos) / 2
    macro_recall = (recall_neg + recall_pos) / 2
    macro_f1 = (f1_neg + f1_pos) / 2
    
    total_support = support_neg + support_pos
    weighted_precision = (precision_neg * support_neg + precision_pos * support_pos) / total_support
    weighted_recall = (recall_neg * support_neg + recall_pos * support_pos) / total_support
    weighted_f1 = (f1_neg * support_neg + f1_pos * support_pos) / total_support

    return {
        'Negative': {'precision': precision_neg, 'recall': recall_neg, 'f1': f1_neg, 'support': support_neg},
        'Positive': {'precision': precision_pos, 'recall': recall_pos, 'f1': f1_pos, 'support': support_pos},
        'accuracy': accuracy,
        'macro avg': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1, 'support': total_support},
        'weighted avg': {'precision': weighted_precision, 'recall': weighted_recall, 'f1': weighted_f1, 'support': total_support}
    }

# Function to load text data from the specified folder
def load_data(folder):
    texts, labels = [], []
    for label, class_name in enumerate(["neg", "pos"]):
        class_folder = os.path.join(folder, class_name)
        for filename in os.listdir(class_folder):
            with open(os.path.join(class_folder, filename), "r", encoding="utf-8") as file:
                texts.append(file.read())
                labels.append(label)
    return texts, labels

# Function to preprocess text: tokenization, stopword removal, feature selection
def preprocess_texts(texts, labels, n, k, m, vocab=None):
    word_counts = Counter()
    doc_counts = Counter()
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        return text.split()
    
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Compute word frequencies and document frequencies
    for words in cleaned_texts:
        word_counts.update(words)
        doc_counts.update(set(words))
    
    # Remove the n most common words and k rare words
    common_words = set([w for w, _ in doc_counts.most_common(n)])
    rare_words = set([w for w, c in doc_counts.items() if c <= k])
    vocab_candidates = list(set(word_counts.keys()) - common_words - rare_words)
    
    # Create initial feature matrix
    X_raw = np.zeros((len(cleaned_texts), len(vocab_candidates)))
    vocab_dict = {word: i for i, word in enumerate(vocab_candidates)}
    
    for i, words in enumerate(cleaned_texts):
        for word in words:
            if word in vocab_dict:
                X_raw[i, vocab_dict[word]] = 1
    
    # Normalize feature matrix
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_raw = scaler.fit_transform(X_raw)
    
    # Select top m words based on mutual information
    from sklearn.feature_selection import mutual_info_classif
    mi_scores = mutual_info_classif(X_raw, np.array(labels), discrete_features=True)
    top_m_indices = np.argsort(mi_scores)[-m:]
    if vocab is None:
        vocab = [vocab_candidates[i] for i in top_m_indices]
    
    # Create the final feature matrix using the selected vocabulary
    vocab_dict = {word: i for i, word in enumerate(vocab)}
    X = np.zeros((len(cleaned_texts), len(vocab)))
    for i, words in enumerate(cleaned_texts):
        for word in words:
            if word in vocab_dict:
                X[i, vocab_dict[word]] = 1
    
    return X, vocab

# Load datasets
train_folder = input("Enter training data folder: ")
test_folder = input("Enter test data folder: ")
n, k, m = map(int, input("Enter n, k, m values: ").split())

train_texts, train_labels = load_data(train_folder)
test_texts, test_labels = load_data(test_folder)
X_train, vocab = preprocess_texts(train_texts, train_labels, n, k, m)
X_test, _ = preprocess_texts(test_texts, test_labels, n, k, m, vocab)
y_train = np.array(train_labels)
y_test = np.array(test_labels)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize learning curves arrays
batch_sizes = np.linspace(0.1, 0.9, 10)  
precision_curve_train, recall_curve_train, f1_curve_train = [], [], []
precision_curve_val, recall_curve_val, f1_curve_val = [], [], []

# Train the model on different fractions of training data and collect learning curve metrics
for batch in batch_sizes:
    X_batch, _, y_batch, _ = train_test_split(X_train, y_train, train_size=float(batch), random_state=42)
    model = SGDClassifier(loss='log_loss', max_iter=500, early_stopping=True, learning_rate='optimal', random_state=42)
    model.fit(X_batch, y_batch)
    
    preds_train = model.predict(X_batch)
    preds_val = model.predict(X_val)
    
    # Here, the provided function "precision_recall_f1" calculates metrics only for the negative class.
    # For learning curves we use that function (as before) for demonstration.
    def precision_recall_f1_neg(y_true, y_pred):
        tp = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 0) & (y_true == 1))
        fn = np.sum((y_pred == 1) & (y_true == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    p_t, r_t, f_t = precision_recall_f1_neg(y_batch, preds_train)
    p_v, r_v, f_v = precision_recall_f1_neg(y_val, preds_val)
    
    precision_curve_train.append(p_t)
    recall_curve_train.append(r_t)
    f1_curve_train.append(f_t)
    
    precision_curve_val.append(p_v)
    recall_curve_val.append(r_v)
    f1_curve_val.append(f_v)

# Plot Precision Learning Curve
plt.figure(figsize=(8, 5))
plt.plot(batch_sizes, precision_curve_train, label='Train Precision (Negative)', marker='o')
plt.plot(batch_sizes, precision_curve_val, label='Validation Precision (Negative)', marker='s')
plt.xlabel('Fraction of Training Data')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision Learning Curve (Negative Class)')
plt.grid(True)
plt.show()

# Plot Recall Learning Curve
plt.figure(figsize=(8, 5))
plt.plot(batch_sizes, recall_curve_train, label='Train Recall (Negative)', marker='o')
plt.plot(batch_sizes, recall_curve_val, label='Validation Recall (Negative)', marker='s')
plt.xlabel('Fraction of Training Data')
plt.ylabel('Recall')
plt.legend()
plt.title('Recall Learning Curve (Negative Class)')
plt.grid(True)
plt.show()

# Plot F1-score Learning Curve
plt.figure(figsize=(8, 5))
plt.plot(batch_sizes, f1_curve_train, label='Train F1-score (Negative)', marker='o')
plt.plot(batch_sizes, f1_curve_val, label='Validation F1-score (Negative)', marker='s')
plt.xlabel('Fraction of Training Data')
plt.ylabel('F1-score')
plt.legend()
plt.title('F1-score Learning Curve (Negative Class)')
plt.grid(True)
plt.show()

# Final evaluation on the Test Set
# Train a final model on the full training set
final_model = SGDClassifier(loss='log_loss', max_iter=500, early_stopping=True, learning_rate='optimal', random_state=42)
final_model.fit(X_train, y_train)
test_preds = final_model.predict(X_test)

# Compute test set metrics for both classes
metrics = compute_metrics(y_test, test_preds)

# Print the classification report in the desired table format
print("              precision    recall  f1-score   support")
print("------------------------------------------------------")
print(f"Negative       {metrics['Negative']['precision']:.2f}      {metrics['Negative']['recall']:.2f}      {metrics['Negative']['f1']:.2f}     {metrics['Negative']['support']}")
print(f"Positive       {metrics['Positive']['precision']:.2f}      {metrics['Positive']['recall']:.2f}      {metrics['Positive']['f1']:.2f}     {metrics['Positive']['support']}")
print()
print(f"accuracy                           {metrics['accuracy']:.2f}     {metrics['Negative']['support'] + metrics['Positive']['support']}")
print(f"macro avg      {metrics['macro avg']['precision']:.2f}      {metrics['macro avg']['recall']:.2f}      {metrics['macro avg']['f1']:.2f}     {metrics['macro avg']['support']}")
print(f"weighted avg   {metrics['weighted avg']['precision']:.2f}      {metrics['weighted avg']['recall']:.2f}      {metrics['weighted avg']['f1']:.2f}     {metrics['weighted avg']['support']}")