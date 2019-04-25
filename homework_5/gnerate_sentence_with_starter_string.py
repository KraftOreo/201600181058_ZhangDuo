import numpy as np
import pickle

with open(r"char-rnn-snapshot.pkl", 'rb')as f:
    a = pickle.load(f, encoding='latin1')
Wxh = a["Wxh"]
Whh = a["Whh"]
Why = a["Why"]
bh = a["bh"]
by = a["by"]
mWxh, mWhh, mWhy = a["mWxh"], a["mWhh"], a["mWhy"]
mbh, mby = a["mbh"], a["mby"]
chars, data_size, vocab_size, char_to_ix, ix_to_char = a["chars"].tolist(), a["data_size"].tolist(), a[
    "vocab_size"].tolist(), a["char_to_ix"].tolist(), a["ix_to_char"].tolist()

data = open('samples.txt', 'r').read()
hidden_size = 250  # size of hidden layer of neurons


def sample(h, seed_ix: list):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """

    for t in range(len(seed_ix)):
        x = np.zeros((vocab_size, 1))
        x[seed_ix[t]] = 1
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    return h


def sample1(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y * 3) / np.sum(np.exp(y * 3))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


# prepare inputs (we're sweeping from left to right in steps seq_length long)


p = 0  # go from start of data
inputs = [char_to_ix[ch] for ch in data]
# sample from the model now and then
h = sample(np.zeros((hidden_size, 1)), inputs)
ixes = sample1(h, inputs[-1], 100)
print(data + ''.join([ix_to_char[i] for i in ixes]))
