import joblib
from Bio import SeqIO

# Load the saved SVM sigmoid model and pre-fitted CountVectorizer
#svm_sigmoid_model = joblib.load('svm_classifier_sigmoid_kernel.pkl')
svm_linear_model = joblib.load('svm_classifier_linear_kernel.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def parse_fasta(fasta_file):
    sequences = []
    headers = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        headers.append(record.id)  # Store the sequence ID (header)
        sequences.append(str(record.seq))  # Store the sequence
    return headers, sequences

fasta_file = 'bacterial_proteins3.fasta'
headers, sequences = parse_fasta(fasta_file)

X_features = vectorizer.transform(sequences)

# Predict the class (0 or 1)
predictions = svm_linear_model.predict(X_features)

# Save positive predictions (class 1) to a text file
with open('ripp_preds2.txt', 'w') as output_file:
    for header, sequence, prediction in zip(headers, sequences, predictions):
        if prediction == 1:  # If the sequence is predicted to be positive
            output_file.write(f'>{header}\n{sequence}\n')

print("Positive predictions saved to ripp_preds2.txt")
