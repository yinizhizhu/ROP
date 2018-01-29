import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from RAN import restorator, discirminator
from patch_wise import patch

from pycrayon import CrayonClient
import time

cc = CrayonClient(hostname="localhost", port=8889)

cc.remove_experiment('d_real_error')
cc.remove_experiment('d_fake_error')
cc.remove_experiment('g_error')
d_real_errorC = cc.create_experiment('d_real_error')
d_fake_errorC = cc.create_experiment('d_fake_error')
g_errorC = cc.create_experiment('g_error')


def extract(v):
    return v.data.storage().tolist()

print 'Starting my Restoration Adversarial Net...'

torch.manual_seed(123)
torch.cuda.manual_seed(123)

patchSize = 64
patches = patch()

# Generator
# G = restorator()
G = torch.load('G.pth')
G = G.cuda()

# Discriminator
# D = discirminator()
D = torch.load('D.pth')
D = D.cuda()

criterionG = nn.MSELoss()
criterionD = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.RMSprop(D.parameters(), lr=1e-4)
g_optimizer = optim.RMSprop(G.parameters(), lr=1e-4)

start = time.time()
for epoch in xrange(600000):
    for k in xrange(5):
        D.zero_grad()

        patches.getNext()
        d_real_data = Variable(patches.real).view(1, -1, patchSize, patchSize).cuda()

        d_real_decision = D(d_real_data)
        # print d_real_decision
        d_real_error = criterionD(d_real_decision, Variable(torch.ones(1)).cuda())  # ones = true
        d_real_error.backward() # compute/store gradients, but don't change params

        d_gen_input = Variable(patches.fake).view(1, -1, patchSize, patchSize).cuda()
        d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = D(d_fake_data)
        d_fake_error = criterionD(d_fake_decision, Variable(torch.zeros(1)).cuda())  # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

    G.zero_grad()

    patches.getNext()
    gen_input = Variable(patches.fake).view(1, -1, patchSize, patchSize).cuda()
    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    g_error = criterionD(dg_fake_decision, Variable(torch.ones(1)).cuda())  # we want to fool, so pretend it's all genuine

    g_error.backward()
    g_optimizer.step()  # Only optimizes G's parameters

    end = - time.time()-start
    d_real_errorC.add_scalar_value('d_real_error', extract(d_real_error)[0], end)
    d_fake_errorC.add_scalar_value('d_fake_error', extract(d_fake_error)[0], end)
    g_errorC.add_scalar_value('g_error', extract(g_error)[0], end)

print 'Storing the gblurConv...'
torch.save(G, 'G1.pth')
torch.save(D, 'D1.pth')
print 'Finished!'
print patches.iteration, patches.moveImg