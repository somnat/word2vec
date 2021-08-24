import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy

'''
u_embedding = Embedding for center word
v_embedding = Embedding for neighbor words
'''

class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension

        self.u_embedding = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embedding = nn.embedding(emb_size, emb_dimension, sparse=True)

        initrange = 1.0 / self.emb_dimension

        init.uniform_(self.u_embedding.weight.data, -initrange, initrange)
        init.constant_(self.v_embedding.weight.data, 0)

    
    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embedding(pos_u)
        emb_v = self.v_embedding(pos_v)
        emb_neg_v = self.v_embedding(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embedding.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' %(len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
    
