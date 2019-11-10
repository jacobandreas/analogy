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
    def __init__(self, surface_forms, lemmas, morphs, refs, n_exemplars, token_dict):
        self.surface_forms = surface_forms
        self.lemmas = lemmas
        self.morphs = morphs
        self.refs = refs
        self.n_exemplars = n_exemplars
        self.token_dict = token_dict

        self.n_tokens = len(token_dict)

        self.reverse_token_dict = {v: k for k, v in self.token_dict.items()}

        ids = list(range(len(self.surface_forms)))
        np.random.shuffle(ids)
        n_train = int(len(ids) * .5)
        self.train_ids = ids[:n_train]
        self.test_ids = ids[n_train:]
        self.train_forms = set(tuple(surface_forms[i]) for i in self.train_ids)
        self.test_forms = set(tuple(surface_forms[i]) for i in self.test_ids)

        self.lemma_to_index = defaultdict(list)
        self.morph_to_index = defaultdict(list)
        self.lemma_and_morph_to_index = dict()
        for i in self.train_ids:
            self.lemma_to_index[self.lemmas[i]].append(i)
            self.morph_to_index[self.morphs[i]].append(i)
            self.lemma_and_morph_to_index[self.lemmas[i], self.morphs[i]] = i

        self.train_groups = defaultdict(list)
        self.test_groups = []
        for i in self.train_ids:
            lemma = self.lemmas[i]
            morph = self.morphs[i]
            for same_lemma in self.lemma_to_index[lemma]:
                if same_lemma == i:
                    continue
                for same_morph in self.morph_to_index[morph]:
                    if same_morph == i:
                        continue
                    key = (self.lemmas[same_morph], self.morphs[same_lemma])
                    if key not in self.lemma_and_morph_to_index:
                        self.test_groups.append((i, same_lemma, same_morph))
                    else:
                        different = self.lemma_and_morph_to_index[key]
                        self.train_groups[i].append((same_lemma, same_morph, different))
        self.usable_train_ids = sorted(list(self.train_groups.keys()))

    def render(self, tokens):
        out = []
        for token in tokens:
            out.append(self.reverse_token_dict[token])
        out = [o for o in out if o not in ("<s>", "</s>")]
        return ("".join(out))

    def origin(self, seq):
        seq = tuple(seq)
        if seq in self.train_forms:
            return "train"
        if seq in self.test_forms:
            return "test"
        if seq in self.refs:
            return "valid"
        return "none"

    def __len__(self):
        return len(self.usable_train_ids)

    def __getitem__(self, i):
        i = self.usable_train_ids[i]
        surf = self.surface_forms[i]
        lemma = self.lemmas[i]
        morph = self.morphs[i]

        ex_ids = self.train_groups[i][np.random.choice(len(self.train_groups[i]))]

        exs = [self.surface_forms[ei] for ei in ex_ids]
        return surf, exs

    def _random_exemplar_group(self):
        if np.random.random() < 0.5:
            lemma = np.random.choice(self.lemmas)
            i1, i2 = np.random.choice(self.lemma_to_index[lemma], size=2)
            morph = self.morphs[i1]
            i3 = np.random.choice(self.morph_to_index[morph])
        else:
            morph = np.random.choice(self.morphs)
            i1, i2 = np.random.choice(self.morph_to_index[morph], size=2)
            lemma = self.lemmas[i1]
            i3 = np.random.choice(self.lemma_to_index[lemma])
        return [self.surface_forms[ei] for ei in [i1, i2, i3]]

    def eval_batch(self, size, extrapolate=False):
        exemplars = []
        while len(exemplars) < size:
            if extrapolate:
                group = self.test_groups[np.random.randint(len(self.test_groups))]
                exemplars.append([self.surface_forms[ei] for ei in group])
            else:
                exemplars.append(self[np.random.randint(len(self))][1])
        return exemplars, batch_seqs_2d(exemplars)

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

    with open(os.path.join(FLAGS.morph_data, "ref.json")) as reader:
        refs = json.load(reader)
        refs = set(tuple(t) for t in refs)

    return MorphDataset(surface_forms, lemmas, morphs, refs, FLAGS.n_exemplars, token_dict)

