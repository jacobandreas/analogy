from absl import flags
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from common import _device

FLAGS = flags.FLAGS

flags.DEFINE_integer("n_embed", 64, "embedding size")
flags.DEFINE_integer("n_hidden", 512, "rnn hidden size")
flags.DEFINE_integer("n_layers", 1, "rnn layers")

def _sample(embed, rnn, predict, init_state, init_token, stop_token, count=1, max_len=40):
    with torch.no_grad():
        out = [[] for _ in range(count)]
        last_state = init_state
        last_token = init_token * torch.ones((1, count), dtype=torch.int64, device=_device())
        for i in range(max_len):
            hidden, next_state = rnn(embed(last_token), last_state)
            probs = F.softmax(predict(hidden), dim=2).detach().cpu().numpy()
            next_token = []
            for j in range(count):
                token = np.random.choice(probs.shape[2], p=probs[0, j, :])
                out[j].append(token)
                next_token.append(token)
            last_state = next_state
            last_token = torch.tensor([next_token], dtype=torch.int64, device=_device())

    out_clean = []
    for seq in out:
        if stop_token in seq:
            seq = seq[:seq.index(stop_token)]
        seq = [t for t in seq if t not in (0, init_token)]
        out_clean.append(seq)

    return out_clean

class Seq(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=dataset.n_tokens,
            embedding_dim=FLAGS.n_embed,
        )
        self.rnn = nn.LSTM(
            input_size=FLAGS.n_embed,
            hidden_size=FLAGS.n_hidden,
            num_layers=FLAGS.n_layers,
        )
        self.predict = nn.Linear(
            in_features=FLAGS.n_hidden,
            out_features=dataset.n_tokens,
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        self.n_tokens = dataset.n_tokens

    def forward(self, target, exemplars):
        del exemplars
        n_seq, n_batch = target.shape
        inp = target[:-1, :]
        out = target[1:, :].view((n_seq - 1) * n_batch)

        embedding = self.embed(inp)
        representation, _ = self.rnn(embedding)
        prediction = self.predict(representation)
        prediction = prediction.view((n_seq - 1) * n_batch, self.n_tokens)
        nlprob = self.loss(prediction, out)
        return nlprob

    def sample(self, exemplars, count):
        init_state = [torch.zeros(1, count, FLAGS.n_hidden), torch.zeros(1, count, FLAGS.n_hidden)]
        init_state = [t.to(_device()) for t in init_state]
        return _sample(self.embed, self.rnn, self.predict, init_state, init_token=1, stop_token=9, count=count)

class Vae(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=dataset.n_tokens,
            embedding_dim=FLAGS.n_embed,
        )
        self.enc_rnn = nn.LSTM(
            input_size=FLAGS.n_embed,
            hidden_size=FLAGS.n_hidden,
            num_layers=FLAGS.n_layers,
        )
        self.dec_rnn = nn.LSTM(
            input_size=FLAGS.n_embed,
            hidden_size=FLAGS.n_hidden,
            num_layers=FLAGS.n_layers,
        )
        self.predict = nn.Linear(
            in_features=FLAGS.n_hidden,
            out_features=dataset.n_tokens,
        )
        self.prior = nn.MSELoss()
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        self.n_tokens = dataset.n_tokens

    def forward(self, target, exemplars):
        del exemplars
        n_seq, n_batch = target.shape
        inp = target[:-1, :]
        out = target[1:, :].view((n_seq - 1) * n_batch)

        enc_embedding = self.embed(target)
        _, (enc_encoding, _) = self.enc_rnn(enc_embedding)

        prior_nlprob = self.prior(enc_encoding, torch.zeros_like(enc_encoding))

        encoding = (enc_encoding, torch.zeros_like(enc_encoding))
        dec_embedding = self.embed(inp)
        dec_representation, _ = self.dec_rnn(dec_embedding, encoding)
        prediction = self.predict(dec_representation)
        prediction = prediction.view((n_seq - 1) * n_batch, self.n_tokens)
        pred_nlprob = self.loss(prediction, out)

        return prior_nlprob + pred_nlprob

class CopyVae(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=dataset.n_tokens,
            embedding_dim=FLAGS.n_embed,
        )
        self.enc_rnn = nn.LSTM(
            input_size=FLAGS.n_embed,
            hidden_size=FLAGS.n_hidden,
            num_layers=FLAGS.n_layers,
        )
        self.dec_rnn = nn.LSTM(
            input_size=FLAGS.n_embed,
            hidden_size=FLAGS.n_hidden,
            num_layers=FLAGS.n_layers,
        )
        self.predict = nn.Linear(
            in_features=FLAGS.n_hidden,
            out_features=dataset.n_tokens,
        )
        self.prior = nn.MSELoss()
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        self.n_tokens = dataset.n_tokens

    def forward(self, target, exemplars):
        n_seq, n_batch = target.shape
        n_ex_seq, _, n_ex = exemplars.shape
        inp = target[:-1, :]
        out = target[1:, :].view((n_seq - 1) * n_batch)

        exemplars = exemplars.view(n_ex_seq, n_batch * n_ex)
        ex_embedding = self.embed(exemplars)
        _, (ex_encoding, _) = self.enc_rnn(ex_embedding)
        ex_encoding = ex_encoding.view(n_batch, n_ex, FLAGS.n_hidden)

        enc_embedding = self.embed(target)
        _, (enc_encoding, _) = self.enc_rnn(enc_embedding)
        prior_nlprob = self.prior(enc_encoding, torch.zeros_like(enc_encoding))

        enc_encoding = enc_encoding.squeeze(0).unsqueeze(1).expand_as(ex_encoding)
        attention_weights = (enc_encoding * ex_encoding).sum(dim=2, keepdim=True)
        attention_weights = F.softmax(attention_weights, dim=1)

        weighted_ex = (ex_encoding * attention_weights.expand_as(ex_encoding)).sum(dim=1).unsqueeze(0)
        encoding = (weighted_ex, torch.zeros_like(weighted_ex))

        prior_nlprob = self.prior(enc_encoding, torch.zeros_like(enc_encoding))
        dec_embedding = self.embed(inp)
        dec_representation, _ = self.dec_rnn(dec_embedding, encoding)
        prediction = self.predict(dec_representation)
        prediction = prediction.view((n_seq - 1) * n_batch, self.n_tokens)
        pred_nlprob = self.loss(prediction, out)

        return prior_nlprob + pred_nlprob
