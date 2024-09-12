import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, \
    roc_auc_score
import joblib
import matplotlib.pyplot as plt
import numpy as np


# Function to read and parse FASTA files
def parse_fasta(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    return sequences


# Load positive and negative sequences
positive_seqs = parse_fasta('positive_data.fasta')
negative_seqs = parse_fasta('RiPP_negative_training_set.fasta')

# Label creation: 1 for positive sequences, 0 for negative sequences
X = positive_seqs + negative_seqs
y = [1] * len(positive_seqs) + [0] * len(negative_seqs)


vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4))
X_features = vectorizer.fit_transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Define the kernel functions to use
#kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernels = ['sigmoid']
# Open a text file to store the metrics
with open('svm_metrics.txt', 'w') as metric_file:
    for kernel in kernels:
        # train the SVM model with the given kernel
        svm_classifier = SVC(kernel=kernel, probability=True)
        svm_classifier.fit(X_train, y_train)

        # save the model
        model_filename = f'svm_classifier_{kernel}_kernel.pkl'
        joblib.dump(svm_classifier, model_filename)
        joblib.dump(vectorizer, 'vectorizer.pkl')

        y_pred = svm_classifier.predict(X_test)
        y_pred_prob = svm_classifier.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        # save metrics to file
        metric_file.write(f"Metrics for SVM with {kernel} kernel:\n")
        metric_file.write(f"ROC AUC: {roc_auc:.4f}\n\n")

        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {kernel} kernel')
        plt.legend(loc="lower right")

        plot_filename = f'roc_auc_{kernel}_kernel.png'
        plt.savefig(plot_filename)
        plt.close()

