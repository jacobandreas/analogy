#!/usr/bin/env python3

from absl import app, flags
import torch
from torch import optim
import torch.utils.data as torch_data

from common import _device
import tasks.morph
import models
import evaluation

FLAGS = flags.FLAGS

flags.DEFINE_enum("task", None, ["morph"], "task name")
flags.DEFINE_enum("model", None, ["seq", "vae", "copy_vae", "analogy"], "model class")
flags.DEFINE_integer("n_batch", 100, "batch size")
flags.DEFINE_integer("n_epochs", 1000, "number of passes over the dataset")

def get_dataset():
    if FLAGS.task == "morph":
        return tasks.morph.load()
    assert False, f"unknown task {FLAGS.task}"

def get_model(dataset):
    if FLAGS.model == "seq":
        return models.Seq(dataset)
    if FLAGS.model == "vae":
        return models.Vae(dataset)
    if FLAGS.model == "copy_vae":
        return models.CopyVae(dataset)
    if FLAGS.model == "analogy":
        return models.Analogy(dataset)
    assert False, f"unknown model {FLAGS.model}"

def main(args):
    dataset = get_dataset()
    model = get_model(dataset).to(_device())
    loader = torch_data.DataLoader(
        dataset,
        batch_size=FLAGS.n_batch,
        collate_fn=dataset.collate,
        shuffle=True,
    )

    opt = optim.Adam(model.parameters(), lr=0.001)
    for i_epoch in range(FLAGS.n_epochs):
        epoch_loss = 0
        for i_batch, (target, exemplars) in enumerate(loader):
            target = torch.tensor(target, device=_device())
            exemplars = torch.tensor(exemplars, device=_device())
            n_seq, n_batch, n_ex = exemplars.shape
            loss = model(target, exemplars, epoch=i_epoch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        #eval_batches = [dataset.eval_batch(100) for _ in range(10)]
        eval_seqs, eval_batch = dataset.eval_batch(100)
        samples = model.sample(torch.tensor(eval_batch, device=_device()), 100)
        evaluation.visualize(eval_seqs[:10], samples[:10], dataset)
        print(evaluation.compute_coverage(samples, dataset))
        print(epoch_loss / i_batch)

if __name__ == "__main__":
    app.run(main)
