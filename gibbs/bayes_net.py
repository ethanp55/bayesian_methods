from copy import copy
from typing import List, Dict, Tuple
import numpy as np


class BayesNetNode(object):
    def __init__(self, name: str) -> None:
        self.name = name

    def get_prob(self) -> float:
        pass

    def get_sample(self, parent_vals: Tuple[str]) -> object:
        pass


class BayesNetDiscreteNode(BayesNetNode):
    def __init__(self, name: str, possible_vals: List[str], conditional_probs: Dict[Tuple, List[float]]) -> None:
        BayesNetNode.__init__(self, name)
        self.name = name
        self.possible_vals = possible_vals
        self.conditional_probs = conditional_probs
        self.parents = []
        self.children = []

    def get_prob(self, var_vals: Dict[str, str]) -> float:
        # Grab the node's previously-sampled value
        my_val = var_vals[self.name]
        my_val_idx = self.possible_vals.index(my_val)

        # Grab any parent values the node depends on
        parent_names = [node.golfer_name for node in self.parents]
        parent_vals_tup = []

        for var_name in var_vals.keys():
            if var_name in parent_names:
                parent_vals_tup.append(var_vals[var_name])

        parent_vals_tup = tuple(parent_vals_tup)

        # Get the probabilities of the node's possible values given the values of its parents
        probs = self.conditional_probs.get(parent_vals_tup, None)

        if probs is None:
            raise Exception('Incorrect parent values')

        # Grab the probability of the previously-sampled value
        prob = probs[my_val_idx]

        return prob

    def get_sample(self, var_vals: Dict[str, str]) -> str:
        # Generate an array of probabilities for each of the node's possible values
        probs_array = []

        for val in self.possible_vals:
            new_var_vals = copy(var_vals)
            new_var_vals[self.name] = val
            prob = 1.0

            # Probability given its parents (first part of Markov blanket)
            prob *= self.get_prob(new_var_vals)

            # Probability for each of its children (second part of Markov blanket)
            for child_node in self.children:
                prob *= child_node.get_prob(new_var_vals)

            probs_array.append(prob)

        # Using the probability array, sample a new value
        probs_array = [prob / sum(probs_array) for prob in probs_array]  # Normalize the array
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

