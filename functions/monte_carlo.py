import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

class MonteCarloSimulator():
    def __init__(self, mean:float, stdev:float) -> None:
        self.mean = mean 
        self.stdev = stdev 
        self.rng = np.random.default_rng()
        self.simulation = None

    def simulate_outstanding(self, initial_amt:float, contribution:float, n_year:int = 20, n_iteration: int = 1000, verbose:bool = True) -> pd.DataFrame:
        column_format = ['year', 'start', 'contribute', 'return', 'gain', 'ending']
        all_df = None
        for i in range(n_iteration):
            starting_balance = initial_amt
            sim_values = []
            for y in range(n_year):
                ret = self.rng.normal(loc = self.mean, scale = self.stdev, size = 1)[0]
                current_gain = (starting_balance + contribution) * ret 
                ending_balance = starting_balance + contribution + current_gain
                sim_values.append([y+1, starting_balance, contribution, ret, current_gain, ending_balance])
                starting_balance = ending_balance 
                
            column_names = ['_'.join([c, str(i+1)]) if c != 'year' else c for c in column_format]
            simulation_df = pd.DataFrame(sim_values, columns = column_names).set_index('year')

            if all_df is None:
                all_df = simulation_df.astype(float)
            else:
                all_df = all_df.merge(simulation_df, left_index=True, right_index=True)
            
            if verbose:
                if (i + 1) % 100 == 0:
                    print(f'Iteration {i+1} is completed, {n_iteration - (i+1)} iterations left')
        return all_df

    def get_stat_values(self, sim_df: pd.DataFrame, percentiles:list = [5, 25, 50, 75, 95], join_to_df:bool = False) -> pd.DataFrame:
        if sim_df is None:
            sim_df = self.simulation

        init_cols = [c for c in sim_df.columns if 'ending' in c]
        ending_balance_df = sim_df[init_cols]
        stat_df = pd.DataFrame(index = ending_balance_df.index)

        # calculate mean, stdev, and median
        stat_df.loc[:, 'mean'] = ending_balance_df[init_cols].mean(axis = 1)
        stat_df.loc[:, 'stdev'] = ending_balance_df[init_cols].std(axis = 1)
        stat_df.loc[:, 'median'] = ending_balance_df[init_cols].median(axis = 1)

        # calculate Nth percentile (5, 25, 50, 75, 95)
        for q in percentiles:
            stat_df.loc[:, f'pct_{q}'] = ending_balance_df[init_cols].apply(lambda x: np.percentile(x, q = q), axis = 1)

        if join_to_df:
            return sim_df.merge(stat_df, left_index=True, right_index=True)
        else:
            return stat_df

    def gen_wealth_path(self, initial_amt:float, contribution:float, n_year:int = 20, n_iteration: int = 1000, percentiles:list = [5, 25, 50, 75, 95], verbose:bool = True):
        simulation = self.simulate_outstanding(initial_amt=initial_amt, contribution=contribution, n_year=n_year, n_iteration=n_iteration, verbose = verbose)
        stat_df = self.get_stat_values(sim_df = simulation, percentiles=percentiles, join_to_df=True)
        return stat_df 
    
    def plot_wealth_path(self, stat_df:pd.DataFrame, colors:list = ['red', 'salmon', 'orange', 'lightgreen', 'green']) -> None:
        pct_cols = [c for c in stat_df if 'pct_' in c]
        if len(pct_cols) != len(colors):
            raise ValueError('number of colors is not equal to number of lines in the plot')
        pct_df = stat_df[pct_cols]
        _, ax = plt.subplots(figsize = (12, 4))

        # backward iteration (to flip the label order)
        for i in range(len(pct_cols) -1, -1, -1):
            _pct = pct_cols[i]
            _color = colors[i]
            ax.plot(pct_df[_pct], color = _color, label = _pct.replace('pct_', 'percentile '))

        ax.set_xlabel('Year')
        ax.set_ylabel('Cumulative Outstanding')
        ax.set_title(f'Monte Carlo Simulation for {pct_df.shape[0]} iterations')
        plt.legend()