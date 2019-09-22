import torch.nn as nn
import torch.nn.functional as f
import torch 

import sys

class Classifier(nn.Module):
  def __init__(self, vocab_size, amount_of_classes, batch_size, device, lstm_dim=100, emb_dim=100):
    super(Classifier, self).__init__()

    self.embeddings = nn.Embedding(vocab_size, emb_dim).to(device)
    self.lstm = nn.LSTM(emb_dim, lstm_dim, num_layers=1, batch_first=True).to(device)
    self.lstm_to_classes = nn.Linear(lstm_dim, amount_of_classes).to(device)
    self.batch_size = batch_size
    self.amount_of_classes = amount_of_classes
    self.device = device

  def forward(self, x, t):
    embedded = self.embeddings(x.to(self.device))
    _, hidden_cell = self.lstm(embedded)
    hidden, _ = hidden_cell
    logits = self.lstm_to_classes(hidden)

    logits_resized = logits.view(self.batch_size, self.amount_of_classes)
    target_resized = t.view(self.batch_size, self.amount_of_classes).to(self.device)

    logp = f.logsigmoid(logits_resized)

    activations = torch.sigmoid(logits_resized)
    correct = torch.eq(activations.round(), target_resized)
    accuracy = correct.sum().item() / float(correct.numel())

    return f.binary_cross_entropy_with_logits(
      logits_resized,
      target_resized
    ), accuracy
    
