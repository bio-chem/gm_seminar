from Bio import Entrez, SeqIO


Entrez.email = "..."


def download_bacterial_proteins(count=100000, min_length=50, max_length=100):
    search_term = f"bacteria[ORGN] AND {min_length}:{max_length}[SLEN]"

    handle = Entrez.esearch(db="protein", term=search_term, retmax=count)
    record = Entrez.read(handle)
    handle.close()

    id_list = record['IdList']

    proteins = []
    for i in range(0, len(id_list), 100):
        id_batch = id_list[i:i + 100]
        handle = Entrez.efetch(db="protein", id=",".join(id_batch), rettype="fasta", retmode="text")
        proteins_batch = list(SeqIO.parse(handle, "fasta"))
        proteins.extend(proteins_batch)
        handle.close()

    return proteins


bacterial_proteins = download_bacterial_proteins()

# Save to a file
with open("bacterial_proteins.fasta", "w") as output_handle:
    SeqIO.write(bacterial_proteins, output_handle, "fasta")

print(f"Downloaded {len(bacterial_proteins)} bacterial proteins.")
