from bayes_net import BayesNet, BayesNetContinuousNode
from distribution import *
from gibbs_met_sampler import GibbsSampler
import matplotlib.pyplot as plt
from scipy.stats import invgamma, norm

N_SAMPLES = 5000
BURN_IN_PERIOD = 50

data = [6.39, 6.32, 6.25, 6.24, 6.21, 6.18, 6.17, 6.13, 6.00, 6.00, 5.97, 5.82, 5.81, 5.71, 5.55, 5.50, 5.39, 5.37,
        5.35, 5.30, 5.27, 4.94, 4.50]

mean_node_mu, mean_node_var = 5.0, 1 / 9
var_node_alpha, var_node_beta = 11.0, 2.5

mean_node = BayesNetContinuousNode('M', NormalDistribution(mean_node_mu, mean_node_var), 0.2 ** 2)
var_node = BayesNetContinuousNode('V', InverseGammaDistribution(var_node_alpha, var_node_beta), 0.15 ** 2)

bayes_net = BayesNet()

bayes_net.add_node(mean_node)
bayes_net.add_node(var_node)

observed_vals = {}

for i in range(len(data)):
    observed_val = data[i]
    name = f'Observation{i + 1}'
    observed_vals[name] = observed_val
    observed_node = BayesNetContinuousNode(name, NormalDistribution(mean_node.name, var_node.name), 0.15 ** 2)
    bayes_net.add_node(observed_node)
    bayes_net.add_edge(mean_node.name, observed_node.name)
    bayes_net.add_edge(var_node.name, observed_node.name)

gibbs = GibbsSampler(bayes_net, N_SAMPLES, BURN_IN_PERIOD)
starting_vals = {mean_node.name: 5.0, var_node.name: 0.3}
simulation_table = gibbs.make_estimate(observed_vals, starting_vals)

# Graphs
mean_node_samples = simulation_table[mean_node.name]
var_node_samples = simulation_table[var_node.name]

prior_samples = 1000

# Mean
plt.hist(norm.rvs(mean_node_mu, mean_node_var ** 0.5, size=prior_samples), int(prior_samples / 5), color='red',
         label='Prior')
plt.hist(mean_node_samples, int(len(mean_node_samples) / 5), color='blue', label='Posterior')
plt.title('Prior and Posterior of Mean')
plt.legend(loc="upper left")
plt.savefig(f'./images/faculty_mean_prior_posterior.png', bbox_inches='tight')
plt.clf()

plt.plot(mean_node_samples)
plt.title('Mixing Plot for Mean')
plt.savefig(f'./images/faculty_mean_mixing.png', bbox_inches='tight')
plt.clf()

# Variance
plt.hist(invgamma.rvs(a=var_node_alpha, scale=var_node_beta, size=prior_samples), int(prior_samples / 5),
         color='red', label='Prior')
plt.hist(var_node_samples, int(len(var_node_samples) / 5), color='blue', label='Posterior')
plt.title('Prior and Posterior of Variance')
plt.legend(loc="upper left")
plt.savefig(f'./images/faculty_variance_prior_posterior.png', bbox_inches='tight')
plt.clf()

plt.plot(var_node_samples)
plt.title('Mixing Plot for Variance')
plt.savefig(f'./images/faculty_variance_mixing.png', bbox_inches='tight')
plt.clf()




