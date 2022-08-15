import matplotlib.pyplot as plt 
import pandas as pd 


COLORS = ['orange', 'lightblue']
SIZE = (12, 4)

def price_compare(df: pd.DataFrame, figsize: tuple = SIZE, colors: list = COLORS, dark_mode: bool = False, date_as_index: bool = True, date_col_name: str = None):
    if not date_as_index:
        if date_col_name is None:
            raise ValueError('Date column is not defined')
        else:
            try:
                df.set_index(date_col_name, inplace = True)
            except ValueError as e:
                raise e

    if dark_mode:
        base_font_color = 'white'
    else:
        base_font_color = 'black'

    fig, ax1 = plt.subplots(figsize = figsize)
    ax1.plot(df.index, df[df.columns[0]], color = colors[0])
    ax1.set_xlabel('Date')
    ax1.set_ylabel(df.columns[0], color = colors[0])
    ax1.tick_params(axis = 'x', labelcolor = base_font_color) 
    ax1.tick_params(axis = 'y', labelcolor = colors[0])

    ax2 = ax1.twinx()
    ax2.plot(df.index, df[df.columns[1]], color = colors[1]) 
    ax2.set_ylabel(df.columns[1], color = colors[1])
    ax2.tick_params(axis = 'y', labelcolor = colors[1]) 

    ax2.set_title(f'Price Comparison: {df.columns[0]} and {df.columns[1]}', color = base_font_color)
    fig.tight_layout()


