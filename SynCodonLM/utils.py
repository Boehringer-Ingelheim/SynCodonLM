def clean_split_sequence(seq):
    seq = seq.upper()
    seq = seq.replace('U', 'T')

    for base in seq:
        if base not in {'A', 'T', 'G', 'C'}:
            raise ValueError(f"Invalid character '{base}' found in sequence. Only A, T, G, C, and U are allowed.")

    spaced_seq = " ".join([seq[i:i+3] for i in range(0, len(seq), 3)])

    return spaced_seq

synonymous_codons = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "C": ["TGT", "TGC"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["TTT", "TTC"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "K": ["AAA", "AAG"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "M": ["ATG"],
    "N": ["AAT", "AAC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "Q": ["CAA", "CAG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "W": ["TGG"],
    "Y": ["TAT", "TAC"],
    "*": ["TAA", "TAG", "TGA"]
}