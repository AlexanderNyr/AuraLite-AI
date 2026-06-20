"""Matplotlib visualization hooks for attention/probabilities/embeddings."""
def plot_token_distribution(ax, probs, tokens=None):
    ax.clear(); ax.bar(range(len(probs)), probs)
    if tokens: ax.set_xticks(range(len(tokens)), tokens, rotation=90)
    ax.set_title("Token probability distribution")

def plot_attention_heatmap(ax, attention):
    ax.clear(); ax.imshow(attention, aspect="auto", cmap="viridis"); ax.set_title("Attention heatmap")
