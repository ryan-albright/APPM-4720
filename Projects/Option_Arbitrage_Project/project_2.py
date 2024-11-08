import pandas as pd
import numpy as np
import scipy.optimize
from datetime import datetime

def organize_data (file_name, expiration_date):
    #chunk of code to extract current and expiration dates in the file
    date = pd.read_csv(file_name, sep = ',',  skiprows = lambda x: x not in [2])
    date = date.columns[0].split()
    month = date[1][:3]
    day = date[2][:-1]
    year = date[3][:-1]
    date = f'{month} {day} {year}'
    curr_date = datetime.strptime(date, "%b %d %Y")
    exp_date = datetime.strptime(expiration_date, "%a %b %d %Y")
    
    df = pd.read_csv(file_name, sep = ',', names = ['Expiration Date', 'Calls', 'C Last Sale', 'C Net', 'C Bid', 'C Ask', 'C Volume', ' C IV', 'C Delta', 'C Gamma', 'C Open Interest', 'Strike', 'Puts', 'P Last Sale', 'P Net', 'P Bid', 'P Ask', 'P Volume', 'P IV','P Delta', 'P Gamma', 'P Open Interest'], skiprows = 4)
    df = df[df['Expiration Date'] == expiration_date]
    
    df_w = df[df['Calls'].apply(lambda x: 'SPXW' in x)].dropna(how='all').dropna(axis=1, how='all')
    df_m = df[df['Calls'].apply(lambda x: 'SPX\d+' in x)].dropna(how='all').dropna(axis=1, how='all')
    
    if df_m.empty:
        print('There are only weekly options available for this expiration date.')
        selected_df = df_w
    elif df_w.empty:
        print('There are only monthly options available for this expiration date.')
        selected_df = df_m
    else:
        decision = input('Do you want to check monthly or weekly options for arbitrage? Enter w for weekly and m for monthly.')    
        if decision == 'w':
            selected_df = df_w
        elif decision == 'm':
            selected_df = df_m
    
    return selected_df, (exp_date - curr_date).days / 365

def linear_solver (df, strike, option_type, position, t):
    # creating c vector
    long_call_prices = df['C Ask']
    short_call_prices = -df['C Bid']
    long_put_prices = df['P Ask']
    short_put_prices = -df['P Bid']
    c = np.concatenate((long_call_prices, short_call_prices, long_put_prices, short_put_prices), axis = 0)
    c = np.append(c, [1000*np.exp(-rfr*t), -1000*np.exp(-rfr*t)])
    
  
    # creating A matrix
    n = len(df['Strike'])
    strikes = df['Strike'].tolist()
    
    long_call = np.zeros((1, n))
    for i in range(n):
        new_row = strikes[:i][::-1] + [0] * (n - i)
        long_call = np.vstack([long_call, new_row])
    long_call = np.vstack([long_call, [1] * n])
    
    short_call = -long_call
    
    long_put = np.array(strikes[:n])
    for i in range(1,n):
        new_row =  [0] * (i) + strikes[:n - i]
        long_put = np.vstack([long_put, new_row])
    long_put = np.vstack([long_put, np.zeros((2, n))]) 
    short_put = -long_put
        
    A_part = np.hstack([long_call, short_call, long_put, short_put])
    long_bond = np.array([1000] * (n + 1) + [0]).reshape(8,1) # long bond column
    short_bond = np.array([-1000] * (n + 1) + [0]).reshape(8,1) # short bond column
    A = np.hstack([A_part, long_bond, short_bond])
    
    #creating b vector
    idx = strikes.index(strike)
    if option_type == 'call':
        slope = 1
        if position == 'long':
            b = long_call[:, idx]
        elif position == 'short':
            b = short_call[:, idx]
    elif option_type == 'put':
        slope = 0
        if position == 'long':
            b = long_put[:, idx]
        elif position == 'short':
            b = short_put[:, idx]
    b = -b # desired position should be opposite our current position
    
    pd.DataFrame(A).to_csv('Projects\Option_Arbitrage_Project\A.csv')
    pd.DataFrame(b).to_csv('Projects\Option_Arbitrage_Project\b.csv')
    pd.DataFrame(c).to_csv('Projects\Option_Arbitrage_Project\c.csv')
    
    result = scipy.optimize.linprog(c, A_eq = A, b_eq = b, bounds= (0, 100))
    np.set_printoptions(suppress = True)
    
    A_col = A.shape[1]
    strike_list = 4 * strikes
    
    result_list = result.x.tolist()
    bool_list = []
    for num in result_list:
        if num > 0.01:
            bool_list.append(True)
        else:
            bool_list.append(False)
                    
    if any(bool_list):
        for n, num in enumerate(result.x):
            if num > 0.01:
                if n <= 5:
                    print(f'Buy {round(result.x[n],2)} calls at strike {strike_list[n]}')
                elif n > 5 and n <= 11:
                    print(f'Sell {round(result.x[n],2)} calls at strike {strike_list[n]}')
                elif n > 11 and n <= 17:
                    print(f'Buy {round(result.x[n],2)} puts at strike {strike_list[n]}')
                elif n > 17 and n <= 23:
                    print(f'Sell {round(result.x[n],2)} puts at strike {strike_list[n]}')
                elif n == 24:
                    if result.x[n] - result.x[n + 1] >= 0:
                        print(f'Buy {round(result.x[n] - result.x[n + 1],2)} zero coupon bonds.')
                    else: 
                        print(f'Sell {-round(result.x[n] - result.x[n + 1],2)} zero coupon bonds.')
                        
    print(f'This portfolio will cost {round(result.fun,2)}')
                    
    return result.x

# Enter the file name and the selected date in these formats
file = 'Projects\Option_Arbitrage_Project\SPX Options ver 1.csv'
expiration_date = 'Fri Dec 17 2021' # this will be the expiration date for both the set of options we are looking at and the 
rfr = 0.005 # enter the risk free rate as a decimal

# Enter the selected option data
input_option_strike = 1000
option_type = 'put' # enter 'put' or 'call'
position = 'long' # enter 'short' or 'long'

df, t = organize_data(file, expiration_date)
linear_solver(df, input_option_strike, option_type, position, t)