from pandas import DataFrame
from bayes_net import BayesNet
import pandas as pd
from typing import Dict


class GibbsSampler:
    def __init__(self, bayes_net: BayesNet, n_samples: int, burn_in_period: int) -> None:
        self.bayes_net = bayes_net
        self.n_samples = n_samples
        self.burn_in_period = burn_in_period

    def make_estimate(self, observed_vals: Dict[str, float], starting_vals=None) -> DataFrame:
        node_names = self.bayes_net.get_node_names()

        for observed_var in observed_vals.keys():
            if observed_var not in node_names:
                raise Exception(f'Observed variable {observed_var} is not in the network')

        table = []
        var_vals = {}

        print('INITIALIZING VALUES')

        for node_name, node in self.bayes_net.nodes.items():
            if node_name in observed_vals:
                var_vals[node_name] = observed_vals[node_name]

            elif starting_vals is not None and node_name in starting_vals:
                var_vals[node_name] = starting_vals[node_name]

            else:
                sampled_val = 0.0

                while not node.distribution.in_support(sampled_val):
                    sampled_val = node.distribution.get_candidate_sample(sampled_val, node.candidate_var)

                var_vals[node_name] = sampled_val

        # Burn in
        print('STARTING BURN IN')
        fifths = int(self.burn_in_period / 5)
        percentage = 20

        for burn_in_sample in range(self.burn_in_period):
            if (burn_in_sample + 1) % fifths == 0:
                print(f'{percentage}% done')
                percentage += 20

            sampled_vals = self.bayes_net.run_simulation(var_vals, observed_vals)
            var_vals = sampled_vals

        # Actual samples that we store
        print('BURN IN FINISHED -> STARTING SAMPLING PROCESS')
        fifths = int(self.n_samples / 5)
        percentage = 20

        for sample_num in range(self.n_samples):
            if (sample_num + 1) % fifths == 0:
                print(f'{percentage}% done')
                percentage += 20

            sampled_vals = self.bayes_net.run_simulation(var_vals, observed_vals)
            new_row = [sampled_vals[node_name] for node_name in node_names]
            table.append(new_row)
            var_vals = sampled_vals

        print('SAMPLING COMPLETED')

        table = pd.DataFrame(table, columns=node_names)
        assert table.shape == (self.n_samples, len(node_names))

        return table


