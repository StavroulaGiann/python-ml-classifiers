import math
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split


# Function to read text files from "pos" and "neg" subfolders.
def load_data(folder_path):
    data, labels = [], []
    for category in ["pos", "neg"]:
        folder = os.path.join(folder_path, category)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist.")
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), "r", encoding="utf-8") as file:
                    data.append(file.read())
                    labels.append(category)
    return data, labels

# Naive Bayes Text Classifier for text classification.
class NaiveBayesTextClassifier:
    def __init__(self, n, m, k, alpha=1.0):
        # Laplace smoothing parameter
        self.alpha = alpha
        self.n = n
        self.m = m
        self.k = k
        # Dictionary to store class probabilities
        self.class_probs = {}
        # Dictionary to store conditional probabilities per class
        self.feature_probs = {}
        # Will hold the CountVectorizer instance with final vocabulary
        self.vectorizer = None
        # The final vocabulary
        self.vocabulary = None

    # Vectorize texts and perform feature selection.
    def preprocess_data(self, X_train, y_train):
        vectorizer = CountVectorizer(binary=True, stop_words='english')
        X_train_vectorized = vectorizer.fit_transform(X_train)
        vocab = np.array(vectorizer.get_feature_names_out())
        doc_freq = np.array((X_train_vectorized > 0).sum(axis=0)).flatten()
        
        # Sort by document frequency (ascending)
        sorted_indices = np.argsort(doc_freq)
        # Remove the k rarest and n most frequent words:
        filtered_indices = sorted_indices[self.k:-self.n] if self.n > 0 else sorted_indices[self.k:]
        selected_vocab = vocab[filtered_indices]
        
        # Re-vectorize using the selected vocabulary
        vectorizer = CountVectorizer(binary=True, stop_words='english', vocabulary=selected_vocab)
        X_train_vectorized = vectorizer.fit_transform(X_train)
        
        # Compute mutual information scores and select top m features
        info_gain_scores = mutual_info_classif(X_train_vectorized, y_train, discrete_features=True)
        top_m_indices = np.argsort(info_gain_scores)[-self.m:]
        final_vocab = selected_vocab[top_m_indices]
        
        self.vectorizer = CountVectorizer(binary=True, vocabulary=final_vocab)
        return self.vectorizer.fit_transform(X_train)
    
    # Train the classifier by computing class and conditional feature probabilities.
    def train(self, X_train, y_train):
        X_train_vectorized = self.preprocess_data(X_train, y_train)
        self.vocabulary = self.vectorizer.get_feature_names_out()
        total_samples = len(y_train)
        class_counts = Counter(y_train)
        
        # Compute class probabilities with Laplace smoothing.
        self.class_probs = {c: (class_counts[c] + self.alpha) / (total_samples + 2 * self.alpha) 
                            for c in class_counts}
        
        # For each class, compute conditional probabilities for each feature.
        for c in class_counts:
            indices = [i for i, label in enumerate(y_train) if label == c]
            class_matrix = X_train_vectorized[indices]
            feature_sums = class_matrix.sum(axis=0) + self.alpha
            self.feature_probs[c] = feature_sums / (len(indices) + 2 * self.alpha)
    
    # Predict the class of each sample using log probabilities.
    def predict(self, X):
        X_vectorized = self.vectorizer.transform(X)
        predictions = []
        for sample in X_vectorized:
            probs = {c: math.log(self.class_probs[c]) for c in self.class_probs}
            for c in self.class_probs:
                for feature_index in sample.nonzero()[1]:
                    probs[c] += math.log(self.feature_probs[c][0, feature_index])
            predictions.append(max(probs, key=probs.get))
        return predictions
    
    # Calculate precision, recall, and F1-score for each class.
    def evaluate(self, y_true, y_pred):
        metrics = {"pos": {}, "neg": {}}
        classes = ["pos", "neg"]
        for target_class in classes:
            tp = sum(1 for y, pred in zip(y_true, y_pred) if y == pred == target_class)
            fp = sum(1 for y, pred in zip(y_true, y_pred) if y != target_class and pred == target_class)
            fn = sum(1 for y, pred in zip(y_true, y_pred) if y == target_class and pred != target_class)
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            metrics[target_class] = {"precision": precision, "recall": recall, "f1": f1}
        return metrics

# Compute overall metrics: accuracy, macro averages, weighted averages.
def compute_overall_metrics(y_true, y_pred, eval_metrics):
    total = len(y_true)
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / total
    
    support_neg = sum(1 for yt in y_true if yt == "neg")
    support_pos = sum(1 for yt in y_true if yt == "pos")
    total_support = support_neg + support_pos
    
    precision_neg = eval_metrics["neg"]["precision"]
    recall_neg = eval_metrics["neg"]["recall"]
    f1_neg = eval_metrics["neg"]["f1"]
    
    precision_pos = eval_metrics["pos"]["precision"]
    recall_pos = eval_metrics["pos"]["recall"]
    f1_pos = eval_metrics["pos"]["f1"]
    
    macro_precision = (precision_neg + precision_pos) / 2
    macro_recall = (recall_neg + recall_pos) / 2
    macro_f1 = (f1_neg + f1_pos) / 2
    
    weighted_precision = (precision_neg * support_neg + precision_pos * support_pos) / total_support
    weighted_recall = (recall_neg * support_neg + recall_pos * support_pos) / total_support
    weighted_f1 = (f1_neg * support_neg + f1_pos * support_pos) / total_support
    
    return {
        "accuracy": accuracy,
        "support_neg": support_neg,
        "support_pos": support_pos,
        "total_support": total_support,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }

# Print a formatted classification report.
def print_classification_report(eval_metrics, overall_metrics):
    print("\n              precision    recall  f1-score   support")
    print("------------------------------------------------------")
    print("Negative       {:.2f}      {:.2f}    {:.2f}     {}".format(
        eval_metrics["neg"]["precision"],
        eval_metrics["neg"]["recall"],
        eval_metrics["neg"]["f1"],
        overall_metrics["support_neg"]
    ))
    print("Positive       {:.2f}      {:.2f}    {:.2f}     {}".format(
        eval_metrics["pos"]["precision"],
        eval_metrics["pos"]["recall"],
        eval_metrics["pos"]["f1"],
        overall_metrics["support_pos"]
    ))
    print("")
    print("accuracy                           {:.2f}     {}".format(
        overall_metrics["accuracy"],
        overall_metrics["total_support"]
    ))
    print("macro avg      {:.2f}      {:.2f}    {:.2f}     {}".format(
        overall_metrics["macro_precision"],
        overall_metrics["macro_recall"],
        overall_metrics["macro_f1"],
        overall_metrics["total_support"]
    ))
    print("weighted avg   {:.2f}      {:.2f}    {:.2f}     {}".format(
        overall_metrics["weighted_precision"],
        overall_metrics["weighted_recall"],
        overall_metrics["weighted_f1"],
        overall_metrics["total_support"]
    ))

# Function to train the classifier on increasing subsets of negative training data and record metrics.
def train_with_learning_curve(classifier, X_train, Y_train, X_dev, Y_dev):
    train_sizes = np.linspace(0.2, 1.0, 5)
    neg_train_precision, neg_train_recall, neg_train_f1 = [], [], []
    neg_dev_precision, neg_dev_recall, neg_dev_f1 = [], [], []

    # Split training data into negative and positive sets.
    X_train_neg = [X_train[i] for i in range(len(Y_train)) if Y_train[i] == "neg"]
    Y_train_neg = ["neg"] * len(X_train_neg)
    X_train_pos = [X_train[i] for i in range(len(Y_train)) if Y_train[i] == "pos"]
    Y_train_pos = ["pos"] * len(X_train_pos)

    # Similarly for development (dev) set.
    X_dev_neg = [X_dev[i] for i in range(len(Y_dev)) if Y_dev[i] == "neg"]
    Y_dev_neg = ["neg"] * len(X_dev_neg)

    for size in train_sizes:
        subset_size = int(len(X_train_neg) * size)
        X_subset_neg = X_train_neg[:subset_size]
        Y_subset_neg = ["neg"] * subset_size
        # Combine with all positive training examples.
        X_subset = X_subset_neg + X_train_pos
        Y_subset = Y_subset_neg + Y_train_pos

        # Train classifier on the subset.
        classifier.train(X_subset, Y_subset)

        # Evaluate on negative training subset.
        Y_pred_train = classifier.predict(X_train_neg[:subset_size])
        train_metrics = classifier.evaluate(Y_train_neg[:subset_size], Y_pred_train)
        neg_train_precision.append(train_metrics["neg"]["precision"])
        neg_train_recall.append(train_metrics["neg"]["recall"])
        neg_train_f1.append(train_metrics["neg"]["f1"])

        # Evaluate on negative development set.
        Y_pred_dev = classifier.predict(X_dev_neg)
        dev_metrics = classifier.evaluate(Y_dev_neg, Y_pred_dev)
        neg_dev_precision.append(dev_metrics["neg"]["precision"])
        neg_dev_recall.append(dev_metrics["neg"]["recall"])
        neg_dev_f1.append(dev_metrics["neg"]["f1"])

    # Multiply train_sizes by total negative training examples to get absolute numbers.
    return train_sizes * len(X_train_neg), neg_train_precision, neg_train_recall, neg_train_f1, neg_dev_precision, neg_dev_recall, neg_dev_f1

# Function to plot learning curves for negative class metrics.
def plot_learning_curves(train_sizes, train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Precision Curve
    axes[0].plot(train_sizes, train_precision, label="Training Precision (neg)", marker='o', linestyle='-')
    axes[0].plot(train_sizes, dev_precision, label="Development Precision (neg)", marker='o', linestyle='--')
    axes[0].set_xlabel("Number of Negative Training Examples")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Learning Curve - Precision (Negative Class)")
    axes[0].legend()
    axes[0].grid()
    axes[0].set_ylim([0, 1])

    # Recall Curve
    axes[1].plot(train_sizes, train_recall, label="Training Recall (neg)", marker='s', linestyle='-')
    axes[1].plot(train_sizes, dev_recall, label="Development Recall (neg)", marker='s', linestyle='--')
    axes[1].set_xlabel("Number of Negative Training Examples")
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Learning Curve - Recall (Negative Class)")
    axes[1].legend()
    axes[1].grid()
    axes[1].set_ylim([0, 1])

    # F1-score Curve
    axes[2].plot(train_sizes, train_f1, label="Training F1-score (neg)", marker='^', linestyle='-')
    axes[2].plot(train_sizes, dev_f1, label="Development F1-score (neg)", marker='^', linestyle='--')
    axes[2].set_xlabel("Number of Negative Training Examples")
    axes[2].set_ylabel("F1-score")
    axes[2].set_title("Learning Curve - F1-score (Negative Class)")
    axes[2].legend()
    axes[2].grid()
    axes[2].set_ylim([0, 1])

    plt.show()

# MAIN
def main():
    train_folder = input("Enter the path to the training data folder: ")
    test_folder = input("Enter the path to the testing data folder: ")
    n, k, m = map(int, input("Enter n, k, m values: ").split())
    
    # Load data.
    X_train, Y_train = load_data(train_folder)
    X_test, Y_test = load_data(test_folder)
    
    # Split training data into training and development sets.
    X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.2, stratify=Y_train, random_state=42)
    
    # Initialize and train classifier on full training data.
    classifier = NaiveBayesTextClassifier(n=n, m=m, k=k, alpha=5.0)
    classifier.train(X_train, Y_train)
    
    # Predict on test set and compute evaluation metrics.
    test_predictions = classifier.predict(X_test)
    eval_metrics = classifier.evaluate(Y_test, test_predictions)
    overall_metrics = compute_overall_metrics(Y_test, test_predictions, eval_metrics)
    
    print("Test Set Evaluation:")
    print_classification_report(eval_metrics, overall_metrics)
    
    # Generate learning curves using increasing negative training data.
    train_sizes, train_prec, train_rec, train_f1, dev_prec, dev_rec, dev_f1 = train_with_learning_curve(
        classifier, X_train, Y_train, X_dev, Y_dev
    )
    plot_learning_curves(train_sizes, train_prec, train_rec, train_f1, dev_prec, dev_rec, dev_f1)
    
if __name__ == "__main__":
    main()