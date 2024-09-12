from Bio import Entrez
from Bio import SeqIO
import random

# Set your email (NCBI requires this)
Entrez.email = "powerbluestz@gmail.com"


# Function to search NCBI and download random nucleotide sequences in FASTA format
def download_random_sequences(query, db="nucleotide", retmax=100000, seq_length_range=(50, 200),
                              output_file="random_sequences.fasta"):
    """
    Downloads random nucleotide sequences from NCBI in FASTA format based on a query.

    Parameters:
    query: str - The search query (e.g., peptides)
    db: str - NCBI database to search (default: nucleotide)
    retmax: int - Maximum number of results to retrieve (default: 10000)
    seq_length_range: tuple - Range of nucleotide sequence lengths to filter (default: 50-150)
    output_file: str - Name of the file to save the downloaded sequences in FASTA format
    """
    try:
        # Search NCBI database with organism filter for Homo sapiens
        search_query = f"{query} AND Homo sapiens[Organism] AND {seq_length_range[0]}:{seq_length_range[1]}[Sequence Length]"
        search_handle = Entrez.esearch(db=db, term=search_query, retmax=retmax, usehistory="y")
        search_results = Entrez.read(search_handle)
        search_handle.close()

        ids = search_results["IdList"]
        if not ids:
            print(f"No results found for query: {query}")
            return

        # Randomly sample 10,000 sequences from the retrieved IDs
        if len(ids) > retmax:
            ids = random.sample(ids, retmax)

        # Fetch the sequences in FASTA format
        fetch_handle = Entrez.efetch(db=db, id=ids, rettype="fasta", retmode="text")
        data = fetch_handle.read()
        fetch_handle.close()

        # Save the fetched sequences to a FASTA file
        with open(output_file, "w") as f:
            f.write(data)

        print(f"Downloaded {len(ids)} random sequences to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage: download 10,000 random sequences for any peptides in humans, 50-200 nucleotides
download_random_sequences("peptide", retmax=10000, output_file="human_peptide_sequences2.fasta")

