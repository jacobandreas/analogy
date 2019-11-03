def visualize(exemplars, samples, dataset):
    for exs, sample in zip(exemplars, samples):
        for ex in exs:
            print(dataset.render(ex))
        print(" ", dataset.render(sample))
        print()

def compute_coverage(samples, dataset):
    counts = {"train": 0, "test": 0, "none": 0}
    samples = set(tuple(s) for s in samples)
    for sample in samples:
        counts[dataset.origin(sample)] += 1
    return counts
