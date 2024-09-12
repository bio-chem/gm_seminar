import pandas as pd

df = pd.read_excel('positve_data.xlsx')

with open('positive_data.fasta', 'w') as fasta_file:
    for index, row in df.iterrows():
        header = f">{row['Compound']}"
        sequence = row['RiPP']
        fasta_file.write(f"{header}\n{sequence}\n")

print("FASTA file created successfully.")
