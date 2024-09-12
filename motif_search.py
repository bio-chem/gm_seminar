from Bio import SeqIO
import re

# Parsing FASTA file
def parse_fasta(file_path):
    sequences = {}
    for record in SeqIO.parse(file_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

# Function to find lanthipeptide motifs
def find_lanthipeptide_motifs(sequence):
    # Regular expression for Cys-Xaa-Cys and Cys-Xaa-Cys-Xaa-Cys motifs
    patterns = [r'C..C..', r'C..C..C']
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, sequence):
            matches.append((pattern, match.start()))
    return matches

fasta_file = "ripp_preds2.txt"
sequences = parse_fasta(fasta_file)

output_file = "lanthipeptide_motifs5.txt"

gene_motifs = {}

# Find motifs and collect gene IDs
for id, seq in sequences.items():
    matches = find_lanthipeptide_motifs(seq)
    if matches:
        if id not in gene_motifs:
            gene_motifs[id] = []
        for pattern, position in matches:
            if pattern not in gene_motifs[id]:
                gene_motifs[id].append(pattern)

with open(output_file, 'w') as f:
    for id, patterns in gene_motifs.items():
        f.write(f"Peptide {id} has the following motifs: {', '.join(patterns)}\n")

