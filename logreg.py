import os
import matplotlib.pyplot as plt
import numpy as np


# Logistic Regression model using Stochastic Gradient Ascent
class LogisticRegressionSGA:
    def __init__(self, lr=0.01, lambda_=0.001, epochs=2000, batch_size=64):
        self.lr = lr
        self.lambda_ = lambda_
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        
        # Store training loss and validation loss if needed
        self.train_loss = []
        self.val_loss = []
        
        # For negative-class metrics, store lists over epochs
        self.train_precision_neg = []
        self.train_recall_neg = []
        self.train_f1_neg = []
        
        self.val_precision_neg = []
        self.val_recall_neg = []
        self.val_f1_neg = []

    # Computes the sigmoid function, which maps any value to the range (0,1).
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Computes the binary cross-entropy loss to measure the model's performance.
    def compute_loss(self, X, y):
        predictions = self.sigmoid(np.dot(X, self.weights))
        loss = -np.mean(
            y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9)
        )
        return loss

    # Compute precision, recall, F1 for the negative class (label=0).
    def compute_negative_metrics(self, X, y):
        # 0 or 1 predictions
        preds = self.predict(X)  
        # True Negatives (TN): y=0, pred=0
        tn = np.sum((y == 0) & (preds == 0))
        # False Positives (FP): y=1, pred=0
        fp = np.sum((y == 1) & (preds == 0))
        # False Negatives (FN): y=0, pred=1
        fn = np.sum((y == 0) & (preds == 1))

        # Precision (Negative) = TN / (TN + FP)
        precision_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        # Recall (Negative) = TN / (TN + FN)
        recall_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        # F1 (Negative)
        if precision_neg + recall_neg > 0:
            f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg)
        else:
            f1_neg = 0.0

        return precision_neg, recall_neg, f1_neg

    #Trains a logistic regression model using mini-batch gradient descent while tracking performance metrics
    def fit(self, X_train, y_train, X_val, y_val):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X_train[indices], y_train[indices]

            for i in range(0, n_samples, self.batch_size):
                xi = X_shuffled[i : i + self.batch_size]
                yi = y_shuffled[i : i + self.batch_size]

                prediction = self.sigmoid(np.dot(xi, self.weights))
                gradient = np.dot(xi.T, (yi - prediction)) / self.batch_size - self.lambda_ * self.weights
                self.weights += self.lr * gradient

            # (Optional) compute train/val loss
            self.train_loss.append(self.compute_loss(X_train, y_train))
            self.val_loss.append(self.compute_loss(X_val, y_val))
            
            # Compute negative-class metrics for training
            p_neg_train, r_neg_train, f_neg_train = self.compute_negative_metrics(X_train, y_train)
            self.train_precision_neg.append(p_neg_train)
            self.train_recall_neg.append(r_neg_train)
            self.train_f1_neg.append(f_neg_train)
            
            # Compute negative-class metrics for validation
            p_neg_val, r_neg_val, f_neg_val = self.compute_negative_metrics(X_val, y_val)
            self.val_precision_neg.append(p_neg_val)
            self.val_recall_neg.append(r_neg_val)
            self.val_f1_neg.append(f_neg_val)

    # Converts sigmoid output into binary class predictions (0 or 1)
    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.weights)) >= 0.5).astype(int)

# Function to load dataset from folder
def load_data(folder):
    texts, labels = [], []
    for label, class_name in enumerate(["neg", "pos"]):
        class_folder = os.path.join(folder, class_name)
        for filename in os.listdir(class_folder):
            with open(os.path.join(class_folder, filename), "r", encoding="utf-8") as file:
                texts.append(file.read())
                labels.append(label)
    return texts, labels

# Function to preprocess texts
def preprocess_texts(texts, labels, n, k, m, vocab=None):
    import re
    from collections import Counter

    word_counts = Counter()
    doc_counts = Counter()

    #This code cleans and preprocesses text data and then converts it into a one-hot encoded feature matrix
    def clean_text(text):
        text = text.lower()
        # Remove special characters and keep only alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        words = text.split()
        return words

    cleaned_texts = [clean_text(text) for text in texts]

    if vocab is None:
        # Build vocabulary based on n, k
        for words in cleaned_texts:
            word_counts.update(words)
            doc_counts.update(set(words))

        common_words = set([w for w, c in doc_counts.most_common(n)])
        rare_words = set([w for w, c in doc_counts.items() if c <= k])
        vocab_words = set(word_counts.keys()) - common_words - rare_words
        # Here, simply take the first m words as a demonstration
        vocab_list = list(vocab_words)[:m]
    else:
        vocab_list = vocab

    vocab_dict = {word: i for i, word in enumerate(vocab_list)}

    X = np.zeros((len(cleaned_texts), len(vocab_list)))
    for i, words in enumerate(cleaned_texts):
        for word in words:
            if word in vocab_dict:
                X[i, vocab_dict[word]] = 1

    return X, np.array(labels), vocab_list if vocab is None else None

#Function to plot separate learning curves for negative-class metrics
def plot_negative_learning_curves(model):
    #Plots 3 separate curves (Precision, Recall, F1) for the negative class in training vs. validation sets across epochs.
  
    epochs = np.arange(1, model.epochs + 1)

    # 1. Precision
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, model.train_precision_neg, label="Train Precision (Neg)", marker='o', markersize=2)
    plt.plot(epochs, model.val_precision_neg, label="Val Precision (Neg)", marker='o', markersize=2, linestyle='--')
    plt.title("Negative Class Precision over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Precision (Negative)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Recall
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, model.train_recall_neg, label="Train Recall (Neg)", marker='s', markersize=2, color='red')
    plt.plot(epochs, model.val_recall_neg, label="Val Recall (Neg)", marker='s', markersize=2, linestyle='--', color='red')
    plt.title("Negative Class Recall over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Recall (Negative)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. F1
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, model.train_f1_neg, label="Train F1 (Neg)", marker='^', markersize=2, color='green')
    plt.plot(epochs, model.val_f1_neg, label="Val F1 (Neg)", marker='^', markersize=2, linestyle='--', color='green')
    plt.title("Negative Class F1 over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1 (Negative)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to compute classification metrics for both classes and overall averages
def compute_metrics(y_true, y_pred):
    # For Positive class (label 1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

    # For Negative class (label 0)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp_neg = np.sum((y_true == 1) & (y_pred == 0))
    fn_neg = np.sum((y_true == 0) & (y_pred == 1))
    precision_neg = tn / (tn + fp_neg) if (tn + fp_neg) > 0 else 0
    recall_neg = tn / (tn + fn_neg) if (tn + fn_neg) > 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0

    # Accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    # Macro averages
    macro_precision = (precision_pos + precision_neg) / 2
    macro_recall = (recall_pos + recall_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2

    # Support counts
    support_neg = np.sum(y_true == 0)
    support_pos = np.sum(y_true == 1)
    total_support = support_neg + support_pos

    # Weighted averages
    weighted_precision = (precision_neg * support_neg + precision_pos * support_pos) / total_support
    weighted_recall = (recall_neg * support_neg + recall_pos * support_pos) / total_support
    weighted_f1 = (f1_neg * support_neg + f1_pos * support_pos) / total_support

    return {
        "Negative": (precision_neg, recall_neg, f1_neg, support_neg),
        "Positive": (precision_pos, recall_pos, f1_pos, support_pos),
        "Accuracy": accuracy,
        "Macro Avg": (macro_precision, macro_recall, macro_f1, total_support),
        "Weighted Avg": (weighted_precision, weighted_recall, weighted_f1, total_support)
    }

# Function to print a formatted classification report table
def print_classification_report(test_metrics):
    print("\n              precision    recall  f1-score   support")
    print("------------------------------------------------------")
    neg = test_metrics["Negative"]
    pos = test_metrics["Positive"]
    print("Negative       {:.2f}      {:.2f}    {:.2f}     {}".format(neg[0], neg[1], neg[2], neg[3]))
    print("Positive       {:.2f}      {:.2f}    {:.2f}     {}".format(pos[0], pos[1], pos[2], pos[3]))
    print("")
    print("accuracy                           {:.2f}     {}".format(test_metrics["Accuracy"], 
                                                                    neg[3] + pos[3]))
    macro = test_metrics["Macro Avg"]
    weighted = test_metrics["Weighted Avg"]
    print("macro avg      {:.2f}      {:.2f}    {:.2f}     {}".format(macro[0], macro[1], macro[2], macro[3]))
    print("weighted avg   {:.2f}      {:.2f}    {:.2f}     {}".format(weighted[0], weighted[1], weighted[2], weighted[3]))

# MAIN
if __name__ == "__main__":
    # Load data
    train_folder = input("Enter training data folder: ")
    test_folder = input("Enter test data folder: ")
    n, k, m = map(int, input("Enter n, k, m values: ").split())

    train_texts, train_labels = load_data(train_folder)
    test_texts, test_labels = load_data(test_folder)

    # Preprocess data
    X_train, y_train, vocab_list = preprocess_texts(train_texts, train_labels, n, k, m)
    X_test, y_test, _ = preprocess_texts(test_texts, test_labels, n, k, m, vocab_list)

    # Create and train the model
    model = LogisticRegressionSGA(lr=0.005, lambda_=0.0005, epochs=2000, batch_size=64)
    model.fit(X_train, y_train, X_test, y_test)

    # Compute predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Compute metrics for test set
    test_metrics = compute_metrics(y_test, test_preds)

    # Print classification report for test set
    print_classification_report(test_metrics)

    # Plot the negative-class learning curves (Precision, Recall, F1)
    plot_negative_learning_curves(model)