#
#
#
import torch


#
#
#
class CBOW(torch.nn.Module):
  def __init__(self, voc, emb):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)

  def forward(self, inpt):
    emb = self.emb(inpt)
    emb = emb.mean(dim=1)
    out = self.ffw(emb)
    return out


#
#
#
class Regressor(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.seq = torch.nn.Sequential(
      torch.nn.Linear(in_features=128, out_features=64),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=64, out_features=32),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=32, out_features=16),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=16, out_features=1),
    )

  def forward(self, inpt):
    out = self.seq(inpt)
    return out


#
#
#
if __name__ == '__main__':
  model = CBOW(128, 8)
  print('CBOW:', model)
  criterion = torch.nn.CrossEntropyLoss()
  inpt = torch.randint(0, 128, (3, 5)) # (batch_size, seq_len)
  trgt = torch.randint(0, 128, (3,))   # (batch_size)
  out = model(inpt)
  loss = criterion(out, trgt)
  print(loss) # ~ ln(1/128) --> 4.852...
