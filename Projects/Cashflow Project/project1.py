import datetime
import numpy as np
import scipy.optimize
import pandas as pd
import QuantLib as ql
from dateutil.relativedelta import relativedelta

# Defining Useful Functions
def dirtyPrice(settlementDate, maturityDate, couponRate, cleanPrice):
    '''
    Parameters: settlementDate is the day the bond is purchased in datetime format 
                maturityDate is the day the bond matures in datetime format 
                couponRate is the annual return of the bond in float format
                cleanPrice is the price given, typically $100
    Returns: dirty price or the price you will actually pay
    '''
    ql.Settings.instance().evaluationDate = settlementDate
    
    schedule = ql.Schedule(ql.Date(1,1,2000), maturityDate, ql.Period(ql.Semiannual), ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, True)

    treasury = ql.FixedRateBond(0, 100, schedule, [couponRate], ql.ActualActual(ql.ActualActual.ISMA, schedule))

    actual_price = cleanPrice + ql.BondFunctions.accruedAmount(treasury) 
    
    return actual_price

def bondCashFlows(settlementDate, maturityDate, couponRate, cleanPrice):
    '''
    Parameters: settlementDate is the day the bond is purchased in ql date format 
                maturityDate is the day the bond matures in ql date format 
                couponRate is the annual return of the bond in float format
    Returns: A 2D array of the date and cashflow amount
    '''
    ql.Settings.instance().evaluationDate = settlementDate
    
    schedule = ql.Schedule(ql.Date(1,1,2000), maturityDate, ql.Period(ql.Semiannual), ql.NullCalendar(), ql.Following, ql.Following, ql.DateGeneration.Backward, True)
    
    treasury = ql.FixedRateBond(0, 100, schedule, [couponRate], ql.ActualActual(ql.ActualActual.ISMA, schedule))

    dates, cfs = [],[]

    cashflows = {}
    cashflows[settlementDate.ISO()] = -dirtyPrice(settlementDate, maturityDate, couponRate, cleanPrice)
    for cf in treasury.cashflows():
        date_iso = cf.date().ISO()
        if date_iso not in cashflows:
            cashflows[date_iso] = cf.amount()
        else:
            cashflows[date_iso] += cf.amount()

    df = pd.DataFrame(cashflows.items(), columns=['Date', 'Cashflow'])
    df = df[df['Date'] >= settlementDate.ISO()]
    return df.reset_index().drop(columns=['index'])

def convert_num_date(date):
    '''
    Parameters: a string date in MM/DD/YYY format
    Returns: a quant lib date format of the date
    '''
    date = datetime.datetime.strptime(date, "%m/%d/%Y")
    return ql.Date(date.day, date.month, date.year)

def convert_abbrev_date(date):
    '''
    Parameters: a date in dd-MMM-YYYY format
    Returns: a string in YYY-MM-DD format
    '''
    date = datetime.datetime.strptime(date, "%d-%b-%Y")
    return date.strftime("%Y-%m-%d")

def clean_data (treasury_file, settlementDate):
    '''
    Parameters: a csv file of treasury data
    Returns: A cleaned dataframe
    '''
    
#     ql.Settings.instance().evaluationDate = settlementDate # This line is newly added due to the error
    
    df = pd.read_csv(treasury_file, names = ('CUSIP', 'Type', 'Rate', 'Maturity Date', 'Call Date', 'Buy', 'Sell', 'EOD'))
    
    df = df.drop(columns = ['Call Date', 'Sell', 'EOD'])
    df = df[~df["Type"].isin(["TIPS", "MARKET BASED FRN"])]
    df = df[df['Buy'] != 0.00]
    df = df.reset_index(drop = True)
    
    # Fix date format
    df['Maturity Date'] = df['Maturity Date'].apply(convert_num_date)
    
    # Add dirty price
    dirty_price_list = []
    for i in range(len(df)):
        dirty_price_list.append(dirtyPrice(settlementDate, df['Maturity Date'][i], df['Rate'][i], df['Buy'][i]))
    df['Dirty Price'] = dirty_price_list
    
    return df

def cashflow_dates (df, settlementDate):
    '''
    Parameters: dataframe of treasuries and the settlement date 
    Returns: a dataframe with date as the row label and each treasury CUSIP with cashflows from each one on the corresponding date
    '''
    result = pd.DataFrame(columns = ['Date'])
    
    for i in range(len(df)):
        cashflows = bondCashFlows (settlementDate, df['Maturity Date'][i], df['Rate'][i], df['Buy'][i])
        
        cashflows.columns = ['Date', df["CUSIP"][i]]
        
        result = pd.merge(result, cashflows, how = 'outer', on = 'Date')
        
    return result

def linear_solver (treasury_df, treasury_cashflows, input_cf_file):
    '''
    Parameters: Takes all possible treasuries in a dataframe, the corresponding cashflows of those treasuries, and a required cashflows file
    Returns: A dataframe with the allocations to each treasury to minimize the cost of meeting these required cashflows
    '''
    CF_deficit = pd.read_csv(input_cf_file).rename(columns={"dates": "Date"})
    dates = CF_deficit.drop("cfs", axis=1)
    dates["Date"] = dates["Date"].apply(convert_abbrev_date)

    CF_deficit["Date"] = CF_deficit["Date"].apply(convert_abbrev_date)
    
    treasury_cashflows = pd.merge(treasury_cashflows, dates, how="outer", on="Date").fillna(0) # collect the cash flow dates for the treasuries and required cashflows in one dataframe
    treasury_cashflows = treasury_cashflows.sort_values(by="Date")
    
    b_values = []
    
    for date in treasury_cashflows["Date"]:
        if date in CF_deficit["Date"].values:
            b_values.append(CF_deficit.loc[CF_deficit['Date'] == date, 'cfs'].values[0])
        else:
            b_values.append(0)
            
    rows = len(treasury_cashflows)

    slacks = np.diag(np.negative(np.ones((rows,)))) + np.diag(np.ones((rows - 1,)), k=-1)

    treasury_cashflows_np = treasury_cashflows.drop('Date', axis = 1).values
    cf_matrix = np.concatenate((treasury_cashflows_np, slacks), axis = 1) 
    c = -cf_matrix[0]
    A = cf_matrix[1:]
    
    b = np.negative(np.array(np.negative(b_values)))[1:] # we do not need the 0 on 2/8/2024 as that is just the settlement date and therefore when we purchased the bonds

    bounds = (0, None)

    result = scipy.optimize.linprog(c, A_eq = A, b_eq = b, bounds=bounds)

    allocation = result.x[:len(treasury_df)] * 100
    CUSIP_array = np.array(treasury_df['CUSIP'])

    return pd.DataFrame({'CUSIP':CUSIP_array, 'Principal': allocation.round(2)})

# Solving the linear system

# Settlement Date
buyDate = ql.Date(8, 2, 2024)

# import treasury data
treasury_df = clean_data ('Projects\Cashflow Project\Inputs\TreasuryPrices7Feb24.csv', buyDate)


# Finding all the cf dates for each treasury
treasury_cfs = cashflow_dates(treasury_df, buyDate)

# Using linear solver to find dataframe of correct treasuries
pd.DataFrame(linear_solver (treasury_df, treasury_cfs, 'Projects\Cashflow Project\Inputs\DatesAndCashFlows2.csv')).to_csv('Projects\Cashflow Project\Outputs\Results.csv', index = False)

