from absl import flags
from collections import defaultdict
import json
import numpy as np
import os
import torch.utils.data as torch_data

FLAGS = flags.FLAGS

flags.DEFINE_string("morph_data", None, "location of morphology data")
flags.DEFINE_integer("n_exemplars", 4, "number of exemplars")

def batch_seqs(seqs):
    max_len = max(len(seq) for seq in seqs)
    data = np.zeros((max_len, len(seqs)), dtype=np.int64)
    for i, seq in enumerate(seqs):
        data[:len(seq), i] = seq
    return data

def batch_seqs_2d(seqs2d):
    max_group, = set(len(seqs) for seqs in seqs2d)
    max_len = max(len(seq) for seqs in seqs2d for seq in seqs)
    data = np.zeros((max_len, len(seqs2d), max_group), dtype=np.int64)
    for i, seqs in enumerate(seqs2d):
        for j, seq in enumerate(seqs):
            data[:len(seq), i, j] = seq
    return data

class MorphDataset(torch_data.Dataset):
    def __init__(self, surface_forms, lemmas, morphs, n_exemplars, token_dict):
        self.surface_forms = surface_forms
        self.lemmas = lemmas
        self.morphs = morphs
        self.n_exemplars = n_exemplars
        self.token_dict = token_dict

        self.n_tokens = len(token_dict)

        self.lemma_to_index = defaultdict(list)
        self.morph_to_index = defaultdict(list)
        for i in range(len(self.surface_forms)):
            self.lemma_to_index[self.lemmas[i]].append(i)
            self.morph_to_index[self.morphs[i]].append(i)

        self.reverse_token_dict = {v: k for k, v in self.token_dict.items()}

        ids = list(range(len(self.surface_forms)))
        np.random.shuffle(ids)
        n_train = int(len(ids) * .9)
        self.train_ids = ids[:n_train]
        self.test_ids = ids[n_train:]

    def render(self, tokens):
        out = []
        for token in tokens:
            out.append(self.reverse_token_dict[token])
        return ("".join(out))

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, i):
        i = self.train_ids[i]
        surf = self.surface_forms[i]
        lemma = self.lemmas[i]
        morph = self.morphs[i]

        lemma_ex_ids = np.random.choice(
            self.lemma_to_index[lemma],
            size=self.n_exemplars//2
        )
        morph_ex_ids = np.random.choice(
            self.morph_to_index[morph],
            size=self.n_exemplars//2
        )
        lemma_exs = [self.surface_forms[li] for li in lemma_ex_ids]
        morph_exs = [self.surface_forms[mi] for mi in morph_ex_ids]

        return surf, lemma_exs + morph_exs

    def collate(self, batch):
        target, exemplars = zip(*batch)
        return (batch_seqs(target), batch_seqs_2d(exemplars))

def load():
    with open(os.path.join(FLAGS.morph_data, "data.json")) as reader:
        data = json.load(reader)
    surface_forms = []
    lemmas = []
    morphs = []
    for datum in data:
        surface_forms.append(datum["surface"])
        lemmas.append(datum["lemma"])
        morphs.append(datum["morph"])

    with open(os.path.join(FLAGS.morph_data, "tokens.json")) as reader:
        token_dict = json.load(reader)

    return MorphDataset(surface_forms, lemmas, morphs, FLAGS.n_exemplars, token_dict)
