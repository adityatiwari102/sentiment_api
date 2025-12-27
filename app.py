from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import re

# --- helpers ---
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower()

def encode(text):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()]

# --- model ---
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # assumes <PAD>=0
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

# --- FastAPI app ---
app = FastAPI()

# ✅ Load checkpoint (model + vocab)
checkpoint = torch.load("sentiment_checkpoint.pth", map_location="cpu")
vocab = checkpoint["vocab"]

model = SentimentLSTM(vocab_size=len(vocab), embed_dim=128, hidden_dim=256, output_dim=2)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# --- request body model ---
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    text = input.text
    encoded = torch.tensor(encode(clean_text(text)), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        output = model(encoded)
        pred = torch.argmax(output, dim=1).item()
    label_map = {0:"Negative", 1:"Positive"}
    return {"sentiment": label_map[pred]}
