import torch
import torch.nn as nn


class DQN(nn.Module):
    input_embeds = 9
    output_embeds = 9
    hidden = 64

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(self.input_embeds, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden,self.hidden),
            nn.ReLU(),
        )

        self.adv = nn.Linear(self.hidden,self.output_embeds)
        self.val = nn.Linear(self.hidden,1)


    def forward(self,x):
        x = self.features(x)
        adv = self.adv(x)
        val = self.val(x)

        return val + adv - adv.mean(1,keepdim=True)