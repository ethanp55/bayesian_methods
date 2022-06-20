from bayes_net import BayesNetDiscreteNode, BayesNet
from gibbs_sampler import GibbsSampler

N_SAMPLES = 10000

burglary_node = BayesNetDiscreteNode('B', ['True', 'False'], {(): [0.001, 1 - 0.001]})
earthquake_node = BayesNetDiscreteNode('E', ['True', 'False'], {(): [0.002, 1 - 0.002]})
alarm_node = BayesNetDiscreteNode('A', ['True', 'False'], {('True', 'True'): [0.95, 1 - 0.95],
                                                           ('True', 'False'): [0.94, 1 - 0.94],
                                                           ('False', 'True'): [0.29, 1 - 0.29],
                                                           ('False', 'False'): [0.001, 1 - 0.001]})
john_node = BayesNetDiscreteNode('J', ['True', 'False'], {('True',): [0.90, 1 - 0.90], ('False',): [0.05, 1 - 0.05]})
mary_node = BayesNetDiscreteNode('M', ['True', 'False'], {('True',): [0.70, 1 - 0.70], ('False',): [0.01, 1 - 0.01]})

bayes_net = BayesNet()
bayes_net.add_node(burglary_node)
bayes_net.add_node(earthquake_node)
bayes_net.add_node(alarm_node)
bayes_net.add_node(john_node)
bayes_net.add_node(mary_node)
bayes_net.add_edge(burglary_node.name, alarm_node.name)
bayes_net.add_edge(earthquake_node.name, alarm_node.name)
bayes_net.add_edge(alarm_node.name, john_node.name)
bayes_net.add_edge(alarm_node.name, mary_node.name)

gibbs = GibbsSampler(bayes_net, N_SAMPLES)

print(f'P(Burglary | JohnCalls=true, MaryCalls=true) = '
      f'{gibbs.make_estimate([burglary_node.name], {john_node.name: "True", mary_node.name: "True"})}')
print(f'P(Alarm | JohnCalls=true, MaryCalls=true) = '
      f'{gibbs.make_estimate([alarm_node.name], {john_node.name: "True", mary_node.name: "True"})}')
print(f'P(Earthquake | JohnCalls=true, MaryCalls=true) = '
      f'{gibbs.make_estimate([earthquake_node.name], {john_node.name: "True", mary_node.name: "True"})}')
print(f'P(Burglary | JohnCalls=false, MaryCalls=false) = '
      f'{gibbs.make_estimate([burglary_node.name], {john_node.name: "False", mary_node.name: "False"})}')
print(f'P(Burglary | JohnCalls=true, MaryCalls=false) = '
      f'{gibbs.make_estimate([burglary_node.name], {john_node.name: "True", mary_node.name: "False"})}')
print(f'P(Burglary | JohnCalls=true) = '
      f'{gibbs.make_estimate([burglary_node.name], {john_node.name: "True"})}')
print(f'P(Burglary | MaryCalls=true) = '
      f'{gibbs.make_estimate([burglary_node.name], {mary_node.name: "True"})}')
