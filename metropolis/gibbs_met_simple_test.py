from bayes_net import BayesNet, BayesNetContinuousNode
from distribution import *
from gibbs_met_sampler import GibbsSampler
import matplotlib.pyplot as plt
from scipy.stats import norm

N_SAMPLES = 5000
BURN_IN_PERIOD = 100

a_mean, a_var = 20.0, 2.0

a_node = BayesNetContinuousNode('A', NormalDistribution(a_mean, a_var), 1.0 ** 2)
b_node = BayesNetContinuousNode('B', InverseGammaDistribution(a_node.name, 3.0), 0.2 ** 2)
c_node = BayesNetContinuousNode('C', GammaDistribution(5.0, a_node.name), 0.2 ** 2)

bayes_net = BayesNet()
bayes_net.add_node(a_node)
bayes_net.add_node(b_node)
bayes_net.add_edge(a_node.name, b_node.name)
bayes_net.add_node(c_node)
bayes_net.add_edge(a_node.name, c_node.name)

gibbs = GibbsSampler(bayes_net, N_SAMPLES, BURN_IN_PERIOD)
simulation_table = gibbs.make_estimate({b_node.name: 3.0, c_node.name: 4.0}, {a_node.name: 10.0})

a_node_samples = simulation_table[a_node.name]

prior_samples = 1000

plt.hist(norm.rvs(a_mean, a_var ** 0.5, size=prior_samples), int(prior_samples / 5), color='red', label='Prior')
plt.hist(a_node_samples, int(len(a_node_samples) / 5), color='blue', label='Posterior')
plt.title('Prior and Posterior of A')
plt.legend(loc="upper left")
plt.savefig(f'./images/a_prior_posterior.png', bbox_inches='tight')
plt.clf()

plt.plot(a_node_samples)
plt.title('Mixing Plot for A')
plt.savefig(f'./images/a_mixing.png', bbox_inches='tight')
plt.clf()
