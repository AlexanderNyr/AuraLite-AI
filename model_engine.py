import os

# ---------------------------------------------------------------------------
# CPU multithreading: use all available cores.
# These environment variables must be set BEFORE importing torch/numpy so the
# underlying OpenMP / MKL backends pick them up.
# ---------------------------------------------------------------------------
_CPU_COUNT = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU_COUNT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Tell PyTorch to use all CPU cores for intra-op and inter-op parallelism.
try:
    torch.set_num_threads(_CPU_COUNT)
    # inter-op must be > 0; guard in case it was already configured.
    torch.set_num_interop_threads(max(1, _CPU_COUNT))
except (RuntimeError, ValueError):
    # set_num_interop_threads raises if called after parallel work has started.
    pass

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, seq_length, d_model, d_ff):
        super(MiniTransformer, self).__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, d_model) * 0.01)
        
        # Self-Attention
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x) + self.pos_embedding # (batch, seq_len, d_model)
        
        # Attention
        q = self.W_q(emb)
        k = self.W_k(emb)
        v = self.W_v(emb)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_model)
        
        # Causal Mask
        mask = torch.tril(torch.ones(self.seq_length, self.seq_length)).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        attn_out = self.W_o(context)
        
        # Residual + Norm
        x = self.ln1(emb + attn_out)
        
        # FF + Residual + Norm
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        
        # Output head (last token)
        return self.head(x[:, -1, :])

class CharDataset(Dataset):
    """Sliding-window character dataset.

    Stores the full encoded text once as a single tensor and produces
    (input_seq, next_char) samples on the fly. This keeps memory low and lets
    the DataLoader fetch/collate batches across multiple worker threads.
    """

    def __init__(self, encoded, seq_length):
        # encoded: 1D LongTensor of token ids for the whole text
        self.data = encoded
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return x, y


class AuraLiteEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_threads = torch.get_num_threads()
        self.model = None
        self.optimizer = None
        self.chars = []
        self.char_to_idx = {}
        self.idx_to_char = {}

    def encode(self, s):
        fallback_idx = self.char_to_idx.get(' ', 0)
        return [self.char_to_idx.get(c, fallback_idx) for c in s]

    def train(self, training_text, params, progress_callback=None, stop_event=None):
        seq_length = params.get('seq_length', 16)
        d_model = params.get('d_model', 32)
        d_ff = params.get('d_ff', 64)
        lr = params.get('lr', 0.01)
        epochs = params.get('epochs', 300)
        batch_size = params.get('batch_size', 64)

        # Vocab
        self.chars = sorted(list(set(training_text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Model initialization
        self.model = MiniTransformer(self.vocab_size, seq_length, d_model, d_ff).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Encode the whole text once (single CPU tensor); the Dataset slices it.
        encoded = torch.tensor(self.encode(training_text), dtype=torch.long)
        dataset = CharDataset(encoded, seq_length)
        if len(dataset) == 0:
            raise ValueError("Training text is too short for the chosen Context Window (seq_length).")

        # Multithreaded data loading. Pinning memory speeds up the host->GPU copy
        # when CUDA is available. Workers are only worth spawning for larger data.
        use_workers = (self.num_threads > 1) and (len(dataset) >= 5000)
        num_workers = self.num_threads if use_workers else 0
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=(self.device.type == 'cuda'),
        )
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = 2

        loader = DataLoader(dataset, **loader_kwargs)
        total_batches = len(loader)

        self.model.train()
        for epoch in range(epochs):
            if stop_event and stop_event.is_set():
                break

            running_loss = 0.0
            seen_batches = 0
            stopped_mid_epoch = False

            for xb, yb in loader:
                if stop_event and stop_event.is_set():
                    stopped_mid_epoch = True
                    break

                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                output = self.model(xb)
                loss = criterion(output, yb)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                seen_batches += 1

            if stopped_mid_epoch:
                break

            if progress_callback:
                avg_loss = running_loss / max(1, seen_batches)
                progress_callback(epoch + 1, epochs, avg_loss)

    def generate(self, start_str, length=50):
        if self.model is None:
            raise ValueError("Train the model first!")
        
        self.model.eval()
        res = start_str
        
        with torch.no_grad():
            for _ in range(length):
                inp = res[-self.model.seq_length:]
                if len(inp) < self.model.seq_length:
                    inp = " " * (self.model.seq_length - len(inp)) + inp
                
                ids = torch.tensor([self.encode(inp)], dtype=torch.long).to(self.device)
                logits = self.model(ids)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
                next_id = np.random.choice(self.vocab_size, p=probs)
                res += self.idx_to_char[next_id]
        return res
