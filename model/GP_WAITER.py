from torch import nn
import torch
class PatchEmbedding(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.conv0=nn.Conv2d(1,1,(61,101),(1,1),(30,50))
        # self.conv0 = nn.Conv2d(1, 1, (90, 101), (1, 4))
        self.conv1 = nn.Conv2d(1, 1, (90, 101), (1, 4))
        self.conv2 = nn.Conv2d(1, 1, (60, 101), (1, 2))
        self.conv3 = nn.Conv2d(1, 1, (1, 101), (1, 1))
        self.conv4 = nn.Conv2d(1, 1, (1, 101), (1, 1))
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()
        self.act3 = nn.Tanh()
        self.batchnorm2d = nn.BatchNorm2d(1)
        self.w=w
        
        
    def forward(self,x):
        x = x.unsqueeze(1)
        l0=torch.mul(self.w,x)
        l1 = self.batchnorm2d(self.conv1(l0))
        l2 = self.act1(self.conv2(l1))
        l3 = self.act2(self.conv3(l2))
        l4=self.act3(self.conv4(l3))
        l4=self.batchnorm2d(l4)
        return l4.squeeze()

class AttentionBlock(nn.Module):
    def __init__(self,embed_size1,embed_size2,num_heads):
        super().__init__()
        self.layer_norm_1=nn.LayerNorm(embed_size1)
        self.attn=nn.MultiheadAttention(embed_size1,num_heads,batch_first=True)
        self.layer_norm_2=nn.LayerNorm(embed_size1)
        self.linear_transformer=nn.Sequential(
            nn.Linear(embed_size1, embed_size2),
            nn.GELU()
        )
        self.linear=nn.Sequential(
            nn.Linear(embed_size1,embed_size2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_size2,embed_size2),
            nn.Dropout(0.3)
        )
    def forward(self,x):    
        x=self.layer_norm_1(x)
        x=x+self.attn(x,x,x)[0]
        weight=self.attn(x,x,x)[1]
        x=self.linear_transformer(x)+self.linear(self.layer_norm_2(x))
        return x


class TModel(nn.Module):
    def __init__(self,embed_size,w,param,num_layers):
        super().__init__()
        self.patchembed=PatchEmbedding(w)
        self.layers=nn.Sequential(*(AttentionBlock(embed_size1=param[i]["embed_size1"],embed_size2=param[i]["embed_size2"],num_heads=param[i]["num_heads"]) for i in range(num_layers)))
        self.out=nn.Sequential(nn.LayerNorm(embed_size),nn.Linear(embed_size,1),nn.GELU())
        self.out_r=nn.Sequential(nn.Conv1d(1,1,20,2),nn.GELU(),nn.Conv1d(1,1,7,1),nn.GELU())
        self.batchnorm1d = nn.BatchNorm1d(1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,x):
        output=self.out(self.layers(self.patchembed_t(x))).squeeze()
        output_final = self.out_r(self.batchnorm1d(output.unsqueeze(1))).squeeze()
        return output_final
