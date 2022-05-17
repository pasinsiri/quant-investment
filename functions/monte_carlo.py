import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


class MonteCarloSimulator():
    def __init__(self, n_year:int = 20, n_iteration:int = 1000) -> None:
        self.n_year = n_year 
        self.n_iteration = n_iteration
        self.rng = np.random.default_rng() 
        self.simulation = None 

    def simulate_outstanding(self, initial_amt:float, contribution:float, mean:float, stdev:float, to_pandas:bool = False):
        all_ending_balances = None
        for i in range(self.n_iteration):
            starting_balance = initial_amt
            accumulated_contribution = 0
            balances = []
            for y in range(self.n_year + 1):
                ret = self.rng.normal(loc = mean, scale = stdev, size = 1)[0]
                if y == 0:
                    # * T0 is the moment we invest the starting balance, so there is no return for starting balance and the contribution amount is zero
                    starting_outstanding = starting_balance
                    contribution_outstanding = 0
                else:
                    starting_outstanding = starting_balance * (1 + ret)
                    if y == 1:
                        # * T1 is the moment we invest the first contribution, so there is no return for such contribution
                        contribution_outstanding = contribution
                    else:
                        contribution_outstanding = contribution * (1 + ret)
                accumulated_contribution += contribution_outstanding
                starting_balance = starting_outstanding
                ending = starting_outstanding + accumulated_contribution
                balances.append([(i+1), y, starting_outstanding, accumulated_contribution, ending])
                # starting_balance = ending

                if to_pandas:
                    column_names = ['iteration', 'month', 'start', 'contribution', 'end']
                    balances = pd.DataFrame(balances, columns = ['_'.join([c, str(i)]) if c != 'month' else c for c in column_names]).set_index('month')
        
            if to_pandas:
                # ? export as pd.DataFrame
                if all_ending_balances is None:
                    all_ending_balances = balances
                else:
                    all_ending_balances = all_ending_balances.merge(balances, left_index = True, right_index = True)
            
            else:
                # ? export as np.array
                if all_ending_balances is None:
                    all_ending_balances = [balances]
                else:
                    all_ending_balances.append(balances)
        if to_pandas:       
            return all_ending_balances
        else:
            return np.array(all_ending_balances)

    def get_stat_values(self, simulation, percentiles:list = [5, 25, 50, 75, 95], to_pandas:bool = False):
        # ? balances order: iteration, year, start, contribution, end
        start_values = simulation[:,:,2]
        contrib_values = simulation[:,:,3]
        start_pcts = np.array([np.apply_along_axis(lambda x: np.percentile(x, q), 0, start_values) for q in percentiles])
        contrib_pcts = np.array([np.apply_along_axis(lambda x: np.percentile(x, q), 0, contrib_values) for q in percentiles])
        pcts = np.concatenate((start_pcts, contrib_pcts), axis = 0).transpose()
        if to_pandas:
            col_names = [f'{keyword}_percentile_{q}' for keyword in ['initial', 'contribution'] for q in percentiles]
            pcts = pd.DataFrame(pcts, columns = col_names)
        
        return pcts

    def gen_wealth_path(self, initial_amt:float, contribution:float, mean:float, stdev:float, percentiles:list = [5, 25, 50, 75, 95]):
        simulation = self.simulate_outstanding(initial_amt=initial_amt, contribution=contribution, mean=mean, stdev=stdev)
        pcts = self.get_stat_values(simulation=simulation, percentiles=percentiles, to_pandas=True)
        return pcts 

    def gen_multiple_wealth_path(self, path:str, initial_amt:float, contribution:float, percentiles:list = [5, 25, 50, 75, 95]):
        # ? load port data 
        df = pd.read_csv(path)

        all_wealth_path = None
        for row in [r[1] for r in df.iterrows()]:
            wealth_path = self.gen_wealth_path(initial_amt=initial_amt, contribution=contribution, mean=row['expected_return'], stdev=row['stdev'], percentiles=percentiles)
            wealth_path.insert(0, 'below_minimum', int(row['below_minimum']))
            wealth_path.insert(0, 'risk_level', int(row['risk_level']))
        
            if all_wealth_path is None:
                all_wealth_path = wealth_path
            else:
                all_wealth_path = all_wealth_path.append(wealth_path, ignore_index = True)
            # all_wealth_path[(row['below_minimum'], row['risk_level'])] = wealth_path
        return all_wealth_path

class MonteCarloSimulator_old():
    def __init__(self, mean:float, stdev:float) -> None:
        self.mean = mean 
        self.stdev = stdev 
        self.rng = np.random.default_rng()
        self.simulation = None

    def simulate_outstanding(self, initial_amt:float, contribution:float, n_year:int = 20, n_iteration: int = 1000, verbose:bool = True) -> pd.DataFrame:
        self.n_iteration = n_iteration
        column_format = ['month', 'start', 'contribute', 'return', 'gain', 'ending']
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
                
            column_names = ['_'.join([c, str(i+1)]) if c != 'month' else c for c in column_format]
            simulation_df = pd.DataFrame(sim_values, columns = column_names).set_index('month')

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
        ax.set_title(f'Monte Carlo Simulation for {self.n_iteration} iterations')
        plt.legend()