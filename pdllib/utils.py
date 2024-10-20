import random

def check_sequence(sequence):
    if len(sequence) < 20 or len(sequence) > 6000:
        return False
    elif set(sequence.upper()) - set('ACGTN') != set():
        return False
    else:
        return True

def load_seqfile(file, batch_size=1, sample=10000000, seed=None):
    seqs = []
    batch = []
    filtered = 0

    if file.endswith(".fa") or file.endswith(".fasta"):
        with open(file) as fi:
            for line in fi:
                if line.startswith(">"):
                    continue
                seq = line.strip().upper()
                if check_sequence(seq):
                    batch.append(seq)
                else:
                    filtered += 1
                if len(batch) == batch_size:
                    seqs.append(batch)
                    batch = []
    else:
        if file.endswith(".csv"):
            sep = ","
        else:
            sep = "\t"
        cnt = 0
        seq_idx = -1
        with open(file) as fi:
            for line in fi:
                info = line.strip().split(sep)
                if cnt == 0:
                    if "sequence" in info:
                        seq_idx = info.index("sequence")
                        continue
                if seq_idx == -1:
                    seq = info[-1].upper()
                    if check_sequence(seq):
                        batch.append(seq)
                else:
                    seq = info[seq_idx].upper()
                    if check_sequence(seq):
                        batch.append(seq)
                    else:
                        filtered += 1
                if len(batch) == batch_size:
                    seqs.append(batch)
                    batch = []
    if filtered > 0:
        print(f"Filtered {filtered} sequence(s) due to unsupported chars or length.")

    len_seqs = sum([len(batch) for batch in seqs])
    if len_seqs > sample:
        seqs = [seq for batch in seqs for seq in batch]
        if seed is not None:
            random.seed(seed)
        random.shuffle(seqs)
        seqs = [seqs[i:min(i+batch_size, sample)] for i in range(0, sample, batch_size)]

    return seqs
