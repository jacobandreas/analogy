#!/usr/bin/env python3

from absl import app, flags
import json

FLAGS = flags.FLAGS

flags.DEFINE_string("surface", None, "read path for surface forms")
flags.DEFINE_string("metadata", None, "read path for form metadata")
flags.DEFINE_string("data_out", "data.json", "write path for encoded dataset")
flags.DEFINE_string("tokens_out", "tokens.json", "write path for token dict")
flags.DEFINE_string("lemmas_out", "lemmas.json", "write path for lemma dict")
flags.DEFINE_string("morphs_out", "morphs.json", "write path for morph dict")

def encode(seq, vocab):
    out = []
    for char in seq:
        if char not in vocab:
            vocab[char] = len(vocab)
        out.append(vocab[char])
    return out

def main(args):
    data = []
    token_dict = {"*pad*": 0}
    lemma_dict = {}
    morph_dict = {}
    with open(FLAGS.surface) as surface_reader, open(FLAGS.metadata) as meta_reader:
        for surface, metadata in zip(surface_reader, meta_reader):
            surface = surface.strip()
            lemma, morph, lang = metadata.strip().split("\t")

            if lang not in ("latin", "portuguese", "italian", "spanish"):
                continue

            surface = ["<s>"] + list(surface) + ["</s>"]
            surface_ids = encode(surface, token_dict)
            lemma_id, = encode([f"{lemma} {lang}"], lemma_dict)
            morph_id, = encode([f"{morph} {lang}"], morph_dict)

            data.append({
                "surface": surface_ids,
                "lemma": lemma_id,
                "morph": morph_id
            })

    write = [
        (data, FLAGS.data_out),
        (token_dict, FLAGS.tokens_out),
        (lemma_dict, FLAGS.lemmas_out),
        (morph_dict, FLAGS.morphs_out),
    ]
    for content, path in write:
        with open(path, "w") as writer:
            json.dump(content, writer)

if __name__ == "__main__":
    app.run(main)
