import numpy as np


def lag_dataset(seqs, back):
    X, Y = [], []
    for i in range(back, seqs.shape[1]):
        X.append(seqs[:, :i - 1])
        Y.append(seqs[:, i])
    return X, Y


def lag_dataset_2(seqs, shapes, ref_back, train_back):
    R, X, Y = [], [], []
    for j in range(ref_back, len(shapes) - 1):
        ref = seqs[:, shapes[j - ref_back]:shapes[j]]
        seq = seqs[:, shapes[j]:shapes[j + 1]]
        for i in range(train_back, seq.shape[1]):
            X.append(seq[:, :i - 1])
            Y.append(seq[:, i])
            R.append(ref)
    return R, X, Y
