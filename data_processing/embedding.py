"""Lazy BiomedBERT loading shared by feature preparation and prompt search."""
import os

import numpy as np
import torch


DEFAULT_EMBED_MODEL = os.environ.get(
    "BIOMEDBERT_MODEL", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)


class BiomedicalEmbedder:
    def __init__(self, model_path=DEFAULT_EMBED_MODEL, device="cpu", max_length=512):
        from transformers import AutoModel, AutoTokenizer
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()

    @torch.no_grad()
    def cls(self, texts):
        vectors = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=self.max_length).to(self.device)
            vectors.append(self.model(**inputs).last_hidden_state[:, 0, :].squeeze(0).cpu().numpy())
        return np.asarray(vectors, dtype=np.float32)

    @torch.no_grad()
    def mean(self, texts, batch_size=4):
        """Preserve the notebook's mean over all token positions, including padding."""
        vectors = []
        for start in range(0, len(texts), batch_size):
            inputs = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
            vectors.append(self.model(**inputs).last_hidden_state.mean(1).cpu().numpy())
        if not vectors:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        return np.concatenate(vectors)
