import numpy as np 
import datetime as dt 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm

class MeanReversionTrading():
    def __init__(self, series, transform = False, save_image = True) -> None:
        self.series = series 
        self.transform = transform 
        self.save_image = save_image
        
        # get pair names
        self.asset1 = self.series.columns[0]
        self.asset2 = self.series.columns[1]
        self.pair_name = '_'.join([c for c in self.series])
        self.spread_calculated = False
        self.set_movement = False
        pass

    def plot_price(self, suffix = ''):
        g = self.series.plot(figsize = (10,3))
        plt.title('Price trends of two assets')
        plt.xlabel('Time') 
        plt.ylabel('Price')
        if self.save_image:
            plt.savefig(f'{self.pair_name}_price_{suffix}.jpg')
        plt.show()
        return 

    def plot_correlation(self, suffix = ''):
        g = sns.jointplot(x = self.asset1, y = self.asset2, data = self.series, color = 'orange')
        corr_coef = stats.pearsonr(self.series[self.asset1], self.series[self.asset2])[0]

        plt.rcParams['axes.titlepad'] = 1
        plt.title(f'Correlation between {self.asset2} and {self.asset1} with coefficient = {np.round(corr_coef, 3)}', y = -0.15, x = -2.8)
        if self.save_image:
            plt.savefig(f'{self.pair_name}_correlation_{suffix}.jpg')
        plt.show()
        return

    def calculate_spread(self):
        # * running regression analysis (OLS) 
        estimator = sm.OLS(self.series[self.asset2], self.series[self.asset1]).fit()

        # * calculate spread
        self.series['hedge_ratio'] = -estimator.params[0] 
        self.series['spread'] = self.series[self.asset2] + (self.series[self.asset1] * self.series['hedge_ratio'])
        self.spread_calculated = True
        return self.series

    def plot_spread(self, suffix = ''):
        if not self.spread_calculated:
            print('Spread is not calculated yet, automatically calculate now')
            self.series = self.calculate_spread(self.series)
        plt.plot(self.series['spread'])
        if self.save_image:
            plt.savefig(f'{self.pair_name}_spread_{suffix}.jpg')
        plt.show()
        return 
    
    def test_adfuller(self, verbose = True):
        if not self.spread_calculated:
            print('Spread is not calculated yet, automatically calculate now')
            self.series = self.calculate_spread(self.series)
        cadf = ts.adfuller(self['spread'])
        if verbose:
            print('Augmented Dickey Fuller test statistic =',cadf[0])
            print('Augmented Dickey Fuller p-value =',cadf[1])
            print('Augmented Dickey Fuller 1%, 5% and 10% test statistics =',cadf[4])
        return cadf

    def test_hurst(self, verbose = True, decimal = 2):
        if not self.spread_calculated:
            print('Spread is not calculated yet, automatically calculate now')
            self.series = self.calculate_spread(self.series)

        """Returns the Hurst Exponent of the time series vector ts"""
        # Create the range of lag values
        lags = range(2, 100)
        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(self.series[lag:].values, self.series[:-lag].values))) for lag in lags]
        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        # Return the Hurst exponent from the polyfit output
        hurst_value = poly[0]*2.0
        if verbose:
            print("Hurst Exponent =",round(hurst_value, decimal))
        return hurst_value

    def calculate_halflife(self, verbose = True):
        if not self.spread_calculated:
            print('Spread is not calculated yet, automatically calculate now')
            self.series = self.calculate_spread(self.series)
    
        # Run OLS regression on spread series and lagged version of itself
        spread_lag = self.series['spread'].shift(1)
        spread_lag.iloc[0] = spread_lag.iloc[1]
        spread_ret = self.series['spread'] - spread_lag
        spread_ret.iloc[0] = spread_ret.iloc[1]
        spread_lag2 = sm.add_constant(spread_lag)
        model = sm.OLS(spread_ret,spread_lag2)
        res = model.fit()
        self.halflife = round(-np.log(2) / res.params[1],0)
        if verbose: 
            print(f'Halflife = {self.halflife}')
        return self.halflife

    def plot_z_score_spread(self, suffix = ''):
        if not self.spread_calculated:
            print('Spread is not calculated yet, automatically calculate now')
            self.series = self.calculate_spread(self.series)
        meanSpread = self.series['spread'].rolling(window=int(self.halflife)).mean()
        stdSpread = self.series['spread'].rolling(window=int(self.halflife)).std()

        self.series['zScore'] = (self.series['spread'] - meanSpread) / stdSpread
            
        self.series['zScore'].plot()
        plt.xticks(rotation = 45)
        plt.title('Z-Score (Normalized Value) of Spread')
        plt.xlabel('Time')
        plt.ylabel('Z-Score')
        if self.save_image:
            plt.savefig(f'{self.pair_name}_z_score_spread_{suffix}.jpg')
        plt.show()        

    def set_movement(self, entryZscore = 2, exitZscore = 0, plot_performance = True, suffix = ''):        
        # ? Movement condition
        # * Long position
        # if the previous action is not long and the current Z-score is less than the negative entry score, perform a long entry
        self.series['long entry'] = ((self.series['zScore'] < - entryZscore) & (self.series['zScore'].shift(1) > - entryZscore))

        # if the previous action is long and the current Z-score is more than the negative entry score, perform a long exit
        self.series['long exit'] = ((self.series['zScore'] > - exitZscore) & (self.series['zScore'].shift(1) < - exitZscore))

        self.series['num units long'] = np.nan 
        self.series.loc[self.series['long entry'],'num units long'] = 1 
        self.series.loc[self.series['long exit'],'num units long'] = 0 
        self.series['num units long'][0] = 0 
        self.series['num units long'] = self.series['num units long'].fillna(method='pad') 

        # * Short position
        # if the previous action is not short and the current Z-score is more than the positive entry score, perform a short entry 
        self.series['short entry'] = ((self.series.zScore >  entryZscore) & (self.series.zScore.shift(1) < entryZscore))
        # if the previous action is short and the current Z-score is less than the positive entry score, perform a short exit
        self.series['short exit'] = ((self.series.zScore < exitZscore) & (self.series.zScore.shift(1) > exitZscore))
        self.series.loc[self.series['short entry'],'num units short'] = -1
        self.series.loc[self.series['short exit'],'num units short'] = 0
        self.series['num units short'][0] = 0
        self.series['num units short'] = self.series['num units short'].fillna(method='pad')
        self.set_movement = True

        self.series['numUnits'] = self.series['num units long'] + self.series['num units short']
        self.series['spread pct ch'] = (self.series['spread'] - self.series['spread'].shift(1)) / ((self.series[self.asset1] * abs(self.series['hedge_ratio'])) + self.series[self.asset2])
        self.series['port rets'] = self.series['spread pct ch'] * self.series['numUnits'].shift(1)
            
        self.series['cum rets'] = self.series['port rets'].cumsum()
        self.series['cum rets'] = self.series['cum rets'] + 1

        if plot_performance:
            # Plot the result
            plt.plot(self.series['cum rets'])
            plt.xlabel('Time')
            plt.ylabel('Cumulative Return') 
            if self.save_image:
                plt.savefig(f'{self.pair_name}_MeanReversion_performance_{suffix}.jpg')
            plt.show()
        return self.series

    def calculate_return(self, trading_days = 365, verbose = True, cagr = True, sharpe_ratio = True):
        if cagr and sharpe_ratio:
            both = True
        else:
            both = False

        if cagr:
            start_val = 1
            end_val = self.series['cum rets'].iat[-1]
                
            start_date = self.series.iloc[0].name
            end_date = self.series.iloc[-1].name
            days = (end_date - start_date).days
                
            CAGR = round(((float(end_val) / float(start_val)) ** (float(trading_days)/days)) - 1,4)
            if verbose:
                print(f'CAGR = {CAGR * 100.0}%')
            if not both:
                return cagr
        
        if sharpe_ratio:
            sharpe = (self.series['port rets'].mean() / self.series['port rets'].std()) * np.sqrt(trading_days) 
            if verbose:
                print(f'Sharpe Ratio = {round(sharpe,2)}')
            if not both:
                return sharpe

        if both:
            return cagr, sharpe
