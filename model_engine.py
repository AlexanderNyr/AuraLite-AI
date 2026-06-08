import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

class AuraLiteEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # Vocab
        self.chars = sorted(list(set(training_text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Model initialization
        self.model = MiniTransformer(self.vocab_size, seq_length, d_model, d_ff).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Data prep
        X_data, Y_data = [], []
        for i in range(len(training_text) - seq_length):
            X_data.append(self.encode(training_text[i:i+seq_length]))
            Y_data.append(self.char_to_idx[training_text[i+seq_length]])
        
        X = torch.tensor(X_data, dtype=torch.long).to(self.device)
        Y = torch.tensor(Y_data, dtype=torch.long).to(self.device)

        self.model.train()
        for epoch in range(epochs):
            if stop_event and stop_event.is_set():
                break
                
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, Y)
            loss.backward()
            self.optimizer.step()

            if progress_callback:
                progress_callback(epoch + 1, epochs, loss.item())

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
