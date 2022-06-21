from copy import copy
from distribution import Distribution
import numpy as np
from typing import Dict


class BayesNetNode(object):
    def __init__(self, name: str, candidate_var: float) -> None:
        self.name = name
        self.candidate_var = candidate_var
        self.parents = []
        self.children = []

    def get_prob(self, var_vals: Dict[str, object]) -> float:
        pass

    def get_sample(self, var_vals: Dict[str, object]) -> object:
        pass


class BayesNetContinuousNode(BayesNetNode):
    def __init__(self, name: str, distribution: Distribution, candidate_var: float) -> None:
        BayesNetNode.__init__(self, name, candidate_var)
        self.distribution = distribution

    def get_prob(self, var_vals: Dict[str, float]) -> float:
        return self.distribution.get_likelihood(self.name, var_vals)

    def get_sample(self, var_vals: Dict[str, float]) -> float:
        prev_sample = var_vals.get(self.name, None)

        if prev_sample is None:
            raise Exception(f'No previously found sample for node {self.name}')

        log_likelihood_prev_sample = self.get_prob(var_vals)

        for child_node in self.children:
            log_likelihood_prev_sample += child_node.get_prob(var_vals)

        candidate_sample = self.distribution.get_candidate_sample(prev_sample, self.candidate_var)
        new_var_vals = copy(var_vals)
        new_var_vals[self.name] = candidate_sample

        log_likelihood_candidate = self.get_prob(new_var_vals)

        for child_node in self.children:
            log_likelihood_candidate += child_node.get_prob(new_var_vals)

        r = log_likelihood_candidate - log_likelihood_prev_sample

        use_candidate = np.log(np.random.random()) < r

        return candidate_sample if use_candidate else prev_sample


class BayesNetBinaryNode(BayesNetNode):
    def __init__(self, name: str, distribution: Distribution, candidate_var: float) -> None:
        BayesNetNode.__init__(self, name, candidate_var)
        self.distribution = distribution
        self.possible_vals = [1, 0]

    def get_prob(self, var_vals: Dict[str, float]) -> float:
        return self.distribution.get_likelihood(self.name, var_vals)

    def get_sample(self, var_vals: Dict[str, float]) -> float:
        probs_array = []

        for val in self.possible_vals:
            new_var_vals = copy(var_vals)
            new_var_vals[self.name] = val
            prob = self.get_prob(new_var_vals)

            for child_node in self.children:
                prob += child_node.get_prob(new_var_vals)

            probs_array.append(np.exp(prob))

        probs_array = [prob / sum(probs_array) for prob in probs_array]
        new_sample = np.random.choice(self.possible_vals, p=probs_array)

        return new_sample


class BayesNet:
    def __init__(self) -> None:
        self.nodes = {}

    def add_node(self, new_node: BayesNetNode) -> None:
        if new_node.name in self.nodes:
            raise Exception('Cannot add existing node to the network')

        self.nodes[new_node.name] = new_node

    def add_edge(self, parent_node_name: str, child_node_name: str) -> None:
        if parent_node_name not in self.nodes or child_node_name not in self.nodes:
            raise Exception('Cannot add edge between non-existent nodes')

        self.nodes[parent_node_name].children.append(self.nodes[child_node_name])
        self.nodes[child_node_name].parents.append(self.nodes[parent_node_name])

    def get_node_names(self):
        return list(self.nodes.keys())

    def run_simulation(self, var_vals: Dict[str, object], observed_vals: Dict[str, object]) -> Dict[str, object]:
        # Iterate through the network and use markov blanket combined with the previously-sampled values and any
        # observed values to generate new samples
        sampled_vals = copy(var_vals)

        for node_name, node in self.nodes.items():
            if node_name in observed_vals:
                sampled_vals[node_name] = observed_vals[node_name]

            else:
                sampled_vals[node_name] = node.get_sample(sampled_vals)

        return sampled_vals

