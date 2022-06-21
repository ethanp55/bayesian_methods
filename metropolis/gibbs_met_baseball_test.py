from bayes_net import BayesNet, BayesNetContinuousNode, BayesNetBinaryNode
from distribution import *
from gibbs_met_sampler import GibbsSampler
import matplotlib.pyplot as plt
from scipy.stats import beta, poisson

N_SAMPLES = 5000
BURN_IN_PERIOD = 100

# Priors
base_alpha, base_beta = 815.0, 1595.0
home_run_lambda = 0.041

base_node = BayesNetContinuousNode('B', BetaDistribution(base_alpha, base_beta), 0.05 ** 2)
home_run_node = BayesNetContinuousNode('HR', PoissonDistribution(home_run_lambda), 0.2 ** 2)

bayes_net = BayesNet()
bayes_net.add_node(base_node)
bayes_net.add_node(home_run_node)

# Add observations from 2022
observations = {}
n_plate_appearances, n_times_on_base, n_home_runs = 198, 51, 1

for i in range(n_plate_appearances):
    new_on_base_node = BayesNetBinaryNode(f'OB{i + 1}', BernoulliDistribution(base_node.name), 0.2 ** 2)
    new_home_run_node = BayesNetBinaryNode(f'HR{i + 1}', BernoulliDistribution(home_run_node.name), 0.2 ** 2)

    if i < n_times_on_base:
        observations[new_on_base_node.name] = 1.0

    else:
        observations[new_on_base_node.name] = 0.0

    if i < n_home_runs:
        observations[new_home_run_node.name] = 1.0

    else:
        observations[new_home_run_node.name] = 0.0

    bayes_net.add_node(new_on_base_node)
    bayes_net.add_edge(base_node.name, new_on_base_node.name)
    bayes_net.add_node(new_home_run_node)
    bayes_net.add_edge(home_run_node.name, new_home_run_node.name)

assert len(bayes_net.nodes) == (n_plate_appearances * 2) + 2

gibbs = GibbsSampler(bayes_net, N_SAMPLES, BURN_IN_PERIOD)
simulation_table = gibbs.make_estimate(observations, {base_node.name: 0.5, home_run_node.name: 0.0})

base_node_samples = simulation_table[base_node.name]
home_run_node_samples = simulation_table[home_run_node.name]

prior_samples = 5000

plt.hist(beta.rvs(a=base_alpha, b=base_beta, size=prior_samples), int(prior_samples / 5), color='red', label='Prior')
plt.hist(base_node_samples, int(len(base_node_samples) / 10), color='blue', label='Posterior')
plt.title('Prior and Posterior of Base Node')
plt.legend(loc="upper left")
plt.savefig(f'./images/base_prior_posterior.png', bbox_inches='tight')
plt.clf()

plt.plot(base_node_samples)
plt.title('Mixing Plot for Base Node')
plt.savefig(f'./images/base_mixing.png', bbox_inches='tight')
plt.clf()

plt.hist(poisson.rvs(mu=home_run_lambda, size=prior_samples), int(prior_samples / 5), color='red', label='Prior')
plt.hist(home_run_node_samples, 5, color='blue', label='Posterior')
plt.title('Prior and Posterior of Home Run Node')
plt.legend(loc="upper left")
plt.savefig(f'./images/home_run_prior_posterior.png', bbox_inches='tight')
plt.clf()

plt.plot(home_run_node_samples)
plt.title('Mixing Plot for Home Run Node')
plt.savefig(f'./images/home_run_mixing.png', bbox_inches='tight')
plt.clf()
