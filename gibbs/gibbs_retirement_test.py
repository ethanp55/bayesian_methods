from bayes_net import BayesNetDiscreteNode, BayesNet
from gibbs_sampler import GibbsSampler

# A goofy network

N_SAMPLES = 10000

craving_node = BayesNetDiscreteNode('C', ['True', 'False'], {(): [0.10, 1 - 0.10]})
fast_food_node = BayesNetDiscreteNode('F', ['True', 'False'], {('True',): [0.75, 1 - 0.75],
                                                               ('False',): [0.07, 1 - 0.07]})
job_node = BayesNetDiscreteNode('J', ['True', 'False'], {(): [0.88, 1 - 0.88]})
money_node = BayesNetDiscreteNode('M', ['True', 'False'], {('True', 'True'): [0.75, 1 - 0.75],
                                                           ('True', 'False'): [0.02, 1 - 0.02],
                                                           ('False', 'True'): [0.96, 1 - 0.96],
                                                           ('False', 'False'): [0.07, 1 - 0.07]})
retire_node = BayesNetDiscreteNode('R', ['True', 'False'], {('True',): [0.98, 1 - 0.98], ('False',): [0.01, 1 - 0.01]})

bayes_net = BayesNet()
bayes_net.add_node(craving_node)
bayes_net.add_node(fast_food_node)
bayes_net.add_node(job_node)
bayes_net.add_node(money_node)
bayes_net.add_node(retire_node)
bayes_net.add_edge(craving_node.name, fast_food_node.name)
bayes_net.add_edge(fast_food_node.name, money_node.name)
bayes_net.add_edge(job_node.name, money_node.name)
bayes_net.add_edge(money_node.name, retire_node.name)

gibbs = GibbsSampler(bayes_net, N_SAMPLES)

print(f'P(Craving | Money=true) = {gibbs.make_estimate([craving_node.name], {money_node.name: "True"})}')
print(f'P(Job | Craving=false, Retire=false) = '
      f'{gibbs.make_estimate([job_node.name], {craving_node.name: "False", retire_node.name: "False"})}')
print(f'P(Money | Retire=true, FastFood=true) = '
      f'{gibbs.make_estimate([money_node.name], {retire_node.name: "True", fast_food_node.name: "True"})}')
