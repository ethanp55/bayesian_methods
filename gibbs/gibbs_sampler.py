from bayes_net import BayesNet
import numpy as np
import pandas as pd
from typing import Dict, List


class GibbsSampler:
    def __init__(self, bayes_net: BayesNet, n_samples: int) -> None:
        self.bayes_net = bayes_net
        self.n_samples = n_samples

    def make_estimate(self, vars_in_question: List[str], observed_vals: Dict[str, object]) -> List[str]:
        node_names = self.bayes_net.get_node_names()

        for var_in_question in vars_in_question:
            if var_in_question not in node_names:
                raise Exception(f'Query variable {var_in_question} is not in the network')

        for observed_var in observed_vals.keys():
            if observed_var not in node_names:
                raise Exception(f'Observed variable {observed_var} is not in the network')

        table = []
        var_vals = {}

        for node_name, node in self.bayes_net.nodes.items():
            if node_name in observed_vals:
                var_vals[node_name] = observed_vals[node_name]

            else:
                var_vals[node_name] = np.random.choice(node.possible_vals)

        for _ in range(self.n_samples):
            sampled_vals = self.bayes_net.run_simulation(var_vals, observed_vals)
            new_row = [sampled_vals[node_name] for node_name in node_names]
            table.append(new_row)
            var_vals = sampled_vals

        table = pd.DataFrame(table, columns=node_names)
        assert table.shape == (self.n_samples, len(node_names))

        estimated_probs = []

        for var_in_question in vars_in_question:
            possible_var_vals = self.bayes_net.nodes[var_in_question].possible_vals

            for possible_val in possible_var_vals:
                n_samples = len(table[table[var_in_question] == possible_val])
                prob = n_samples / self.n_samples
                estimated_probs.append(f'{var_in_question} = {possible_val}: {prob}')

        return estimated_probs


