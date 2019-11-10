def visualize(exemplars, samples, dataset):
    for exs, sample in zip(exemplars, samples):
        out = []
        for ex in exs:
            r = dataset.render(ex)
            out.append(f"{r:30s}")
        origin = dataset.origin(tuple(sample))
        if origin == "train":
            sym = " "
        elif origin == "test":
            sym = "X"
        elif origin == "valid":
            sym = "x"
        else:
            sym = "*"
        r = sym + " " + dataset.render(sample)
        out.append(f"{r:30s}")
        print("".join(out))

def compute_coverage(samples, dataset):
    counts = {"train": 0, "test": 0, "valid": 0, "none": 0}
    samples = set(tuple(s) for s in samples)
    for sample in samples:
        counts[dataset.origin(sample)] += 1
    return counts
