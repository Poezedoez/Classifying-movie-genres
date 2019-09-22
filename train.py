import argparse 

import torch
from data_handler import DataHandler
from Classifier import Classifier

def epoch_iter(model, data_handler, device, optimizer, data_type):
  loss = 0.0
  acc = 0.0
  #TODO: fix this for validation data and test data
  for _ in range(data_handler.batches_in_epoch):
    batch, targets = data_handler.load_batch(data_type)

    batch.to(device)
    if model.training:
        optimizer.zero_grad()
    temp_loss, temp_acc = model(batch, targets)

    if model.training:
        temp_loss.backward()
        optimizer.step()
    loss += temp_loss.item()

    acc += temp_acc
  return loss / data_handler.batches_in_epoch, acc / data_handler.batches_in_epoch

def run_epoch(model, data_handler, device, optimizer):
  model.train()
  train_loss, train_acc = epoch_iter(model, data_handler, device, optimizer, "train")

  model.eval()
  with torch.no_grad():
    val_loss, val_acc = epoch_iter(model, data_handler, device, optimizer, "val")
  
  return train_loss, val_loss, train_acc, val_acc

def main(ARGS, device):
  data_handler = DataHandler('data/movies_genres.csv', (0.8, 0.1, 0.1), 32)

  model = Classifier(len(data_handler.vocab), 28, data_handler.batch_size, device)
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters())

  for epoch in range(ARGS.epochs):
    epoch_results = run_epoch(model, data_handler, device, optimizer)
    train_loss, val_loss, train_acc, val_acc = epoch_results
    print(f"[Epoch {epoch} train_loss: {train_loss}, val_loss: {val_loss}, train_acc: {train_acc}, val_acc: {val_acc}]")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', default=1, type=int,
                      help='max number of epochs')
  parser.add_argument('--batch_size', default=5, type=int,
                      help='batch size')
  parser.add_argument('--device', default='cpu', type=str,
                      help='device')

  ARGS = parser.parse_args()
  device = torch.device(ARGS.device)
  main(ARGS, device)
