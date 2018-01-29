import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# from pycrayon import CrayonClient
# import time
#
# cc = CrayonClient(hostname="localhost", port=8889)
#
# cc.remove_experiment('d_real_error')
# cc.remove_experiment('d_fake_error')
# cc.remove_experiment('g_error')
# d_real_errorC = cc.create_experiment('d_real_error')
# d_fake_errorC = cc.create_experiment('d_fake_error')
# g_errorC = cc.create_experiment('g_error')


def d_sampler(n):
    return torch.Tensor(np.random.normal(4, 1.25, (1, n)))  # Gaussian


def gi_sampler(m, n):
    return torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian


class Generator(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return self.map3(x)


class Discriminator(nn.Module):
    def __init__(self, input_size=200, hidden_size=50, output_size=1):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


def preprocess(data):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), 2.0)
    return torch.cat([data, diffs], 1)

print 'Using data [Data and variances]'

G = Generator()
D = Discriminator()

criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=2e-4)
g_optimizer = optim.Adam(G.parameters(), lr=2e-4)

# start = time.time()
for epoch in xrange(1):
    for d_index in xrange(1):
        D.zero_grad()

        d_real_data = Variable(d_sampler(100))
        print d_real_data

        d_real_decision = D(preprocess(d_real_data))
        d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
        d_real_error.backward() # compute/store gradients, but don't change params

        d_gen_input = Variable(gi_sampler(100, 1))
        d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = D(preprocess(d_fake_data.t()))
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

    for g_index in xrange(1):
        G.zero_grad()

        gen_input = Variable(gi_sampler(100, 1))
        g_fake_data = G(gen_input)
        dg_fake_decision = D(preprocess(g_fake_data.t()))
        g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))  # we want to fool, so pretend it's all genuine

        g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters

    # end = - time.time()-start
    # d_real_errorC.add_scalar_value('d_real_error', extract(d_real_error)[0], end)
    # d_fake_errorC.add_scalar_value('d_fake_error', extract(d_fake_error)[0], end)
    # g_errorC.add_scalar_value('g_error', extract(g_error)[0], end)
    print("%s: (Real: %s, Fake: %s) " % (epoch, stats(extract(d_real_data)), stats(extract(d_fake_data))))