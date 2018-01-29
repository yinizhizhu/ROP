import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(1)

PAINT_POINTS = np.vstack([np.linspace(-1,1,15) for _ in range(64)])

def artist_works():
    a = np.random.uniform(1,2,size=64)[:,np.newaxis]
    paintings = a*np.power(PAINT_POINTS,2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return Variable(paintings)

G = nn.Sequential(
    nn.Linear(5,128),
    nn.ReLU(),
    nn.Linear(128, 15),
)

D = nn.Sequential(
    nn.Linear(15,128),
    nn.ReLU(),
    nn.Linear(128,1),
    nn.Sigmoid(),
)

opt_D = torch.optim.Adam(D.parameters(),lr=0.0001)
opt_G = torch.optim.Adam(G.parameters(),lr=0.0001)

plt.ion()

for step in range(1):
    artist_paintings = artist_works()
    prob_artist0 = D(artist_paintings)

    G_ideas = Variable(torch.randn(64, 5))
    G_paintings = G(G_ideas)
    prob_artist1 = D(G_paintings)

    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1-prob_artist1))
    G_loss = torch.mean(torch.log(1 - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_variables=True)
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 50 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0],G_paintings.data.numpy()[0],c='#4ad631',lw=3,label='Generated painting',)
        plt.plot(PAINT_POINTS[0],2 * np.power(PAINT_POINTS[0], 2) + 1,c='#74BCFF',lw=3,label='upper bound',)
        plt.plot(PAINT_POINTS[0],1 * np.power(PAINT_POINTS[0], 2) + 0,c='#FF9359',lw=3,label='lower bound',)
        plt.text(-.5,2.3,'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size':15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 15})
        plt.ylim((0,3))
        plt.legend(loc='upper right', fontsize=12)
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()