import torch
from torch import nn
from torch.nn import functional as F
from model.attention import SelfAttention
class CLIPEmbedding(nn.Module):
    def __init__(self,n_vocab:int, n_embed:int, n_tokens:int):
        super().__init__()
        self.token_embedding=nn.Embedding(n_vocab,n_embed)
        self.position_embedding=nn.Parameter(torch.zeros(n_tokens,n_embed))
        
    def forward(self, tokens):
        x= self.token_embedding(tokens)
        x+=self.position_embedding
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_heads:int, n_embed:int):
        super().__init__()
        
        self.layernorm1=nn.LayerNorm(n_embed)
        self.attention=SelfAttention(n_heads, n_embed)
        self.layernorm2=nn.LayerNorm(n_embed)
        self.linear1=nn.Linear(n_embed, 4*n_embed)
        self.linear2=nn.Linear(4*n_embed, n_embed)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue=x
        x=self.layernorm1(x)
        x=self.attention(x,casual_mask=True)
        x+=residue
        residue=x
        x=self.layernorm2(x)
        x=self.linear1(x)
        x=x*torch.sigmoid(1.702*x)#quick gelu fucntion
        x=self.linear2(x)
        x+=residue
        return x
class CLIP(nn.Module):
    def __init__(self):
        self.embedding= CLIPEmbedding(49408,768,77)

        self.layers=nn.Module([
            CLIPLayer(12,768) for i in range(12)
        ])
        self.layernorm=nn.LayerNorm(768)
    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        tokens=tokens.type(torch.long)
        state=self.embedding(tokens)
        for layer in self.layers:
            state=layer(state)
        output=self.layernorm(state)
        return output