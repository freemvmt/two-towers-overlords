#
#
#
import torch
import pickle


#
#
#
vocab_to_int = pickle.load(open('./data/tkn_words_to_ids.pkl', 'rb'))
int_to_vocab = pickle.load(open('./data/tkn_ids_to_words.pkl', 'rb'))


#
#
#
def topk(mFoo, vocab = 'computer'):

  idx = vocab_to_int.get(vocab, 0)
  vec = mFoo.emb.weight[idx].detach()
  with torch.no_grad():

    vec = torch.nn.functional.normalize(vec.unsqueeze(0), p=2, dim=1)
    emb = torch.nn.functional.normalize(mFoo.emb.weight.detach(), p=2, dim=1)
    sim = torch.matmul(emb, vec.squeeze())
    top_val, top_idx = torch.topk(sim, 6)
    print(f'\nTop 5 words similar to "{vocab}":')
    count = 0
    for i, idx in enumerate(top_idx):
      word = int_to_vocab[idx.item()]
      sim = top_val[i].item()
      print(f'  {word}: {sim:.4f}')
      count += 1
      if count == 5: break
