#!/usr/bin/env python3

from absl import app, flags
import json

FLAGS = flags.FLAGS

flags.DEFINE_string("langs", "language_codes.tsv", "read path for language codes")
flags.DEFINE_string("surface", None, "read path for surface forms")
flags.DEFINE_string("metadata", None, "read path for form metadata")
flags.DEFINE_string("ref_surface", None, "TODO")
flags.DEFINE_string("ref_metadata", None, "TODO")
flags.DEFINE_string("data_out", "data.json", "write path for encoded dataset")
flags.DEFINE_string("tokens_out", "tokens.json", "write path for token dict")
flags.DEFINE_string("lemmas_out", "lemmas.json", "write path for lemma dict")
flags.DEFINE_string("morphs_out", "morphs.json", "write path for morph dict")
flags.DEFINE_string("ref_out", "ref.json", "TODO")

def encode(seq, vocab):
    out = []
    for char in seq:
        if char not in vocab:
            vocab[char] = len(vocab)
        out.append(vocab[char])
    return out

def encode_surface(surface, lang, token_dict):
    surface = [lang + ":"] + list(surface)
    surface = ["<s>"] + surface + ["</s>"]
    return encode(surface, token_dict)

def main(args):
    lang_dict = {}
    with open(FLAGS.langs) as lang_reader:
        for line in lang_reader:
            lang, code = line.strip().split("\t")
            lang_dict[lang] = code
    codes = set(lang_dict.values())

    data = []
    token_dict = {"*pad*": 0}
    lemma_dict = {}
    morph_dict = {}
    with open(FLAGS.surface) as surface_reader, open(FLAGS.metadata) as meta_reader:
        for surface, metadata in zip(surface_reader, meta_reader):
            surface = surface.strip()
            lemma, morph, lang = metadata.strip().split("\t")

            if lang not in ("latin", "portuguese", "italian", "spanish", "galician"):
                continue

            assert lang in lang_dict
            lang = lang_dict[lang]

            surface_ids = encode_surface(surface, lang, token_dict)
            lemma_id, = encode([f"{lemma} {lang}"], lemma_dict)
            morph_id, = encode([f"{morph} {lang}"], morph_dict)

            data.append({
                "surface": surface_ids,
                "lemma": lemma_id,
                "morph": morph_id
            })

    ref_data = []
    with open(FLAGS.ref_surface) as surface_reader, open(FLAGS.ref_metadata) as meta_reader:
        for surface, metadata in zip(surface_reader, meta_reader):
            surface = surface.strip()
            lang = metadata.strip()
            assert lang in codes
            surface_ids = encode_surface(surface, lang, token_dict)
            ref_data.append(surface_ids)

    write = [
        (data, FLAGS.data_out),
        (token_dict, FLAGS.tokens_out),
        (lemma_dict, FLAGS.lemmas_out),
        (morph_dict, FLAGS.morphs_out),
        (ref_data, FLAGS.ref_out),
    ]
    for content, path in write:
        with open(path, "w") as writer:
            json.dump(content, writer)

if __name__ == "__main__":
    app.run(main)
