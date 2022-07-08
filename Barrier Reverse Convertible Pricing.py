# Multi Barrier Reverse Convertible Pricing (assumes dirty quotation incl accrued interest)

import datetime
import pandas_market_calendars as mcal
import numpy as np
import pdblp
from math import sqrt, exp, isnan
from statistics import mean

#####################################
### Input product-specific Parameters

underlyings = ['NESN SE Equity', 'NOVN SE Equity', 'ROG SE Equity'] # Underlyings
IA = .04 # Annual Interest Amount
Ks = [115.62, 84.57, 347.3] # Strikes
barrier = .6 # Barrier Level
SD = 1000 # Specified Denomination / Investment
idate = datetime.date(2021, 7, 28) # Issue Date (YYYY, MM, DD)
vdate = datetime.date(2024, 7, 23) # Valuation Date
cfdates = [datetime.date(2021, 7, 28),
           datetime.date(2021, 11, 1), datetime.date(2022, 1, 31), datetime.date(2022, 1, 31),
           datetime.date(2022, 5, 2), datetime.date(2022, 8, 2), datetime.date(2022, 10, 31),
           datetime.date(2023, 1, 30), datetime.date(2023, 5, 2), datetime.date(2023, 7, 31),
           datetime.date(2023, 10, 30), datetime.date(2024, 1, 30), datetime.date(2024, 4, 30),
           datetime.date(2024, 7, 30)] # CF dates (first element: start date, middle elements: coupons, last element: principal & coupon)

#####################################
#DON'T EDIT BELOW THIS LINE
#####################################

# BBG connection
con = pdblp.BCon(debug = False, port = 8194, timeout = 5000)
con.start()
con.debug = False
con.start()

# Parameters
N = 1000 # Number of Simulations
fre = '1m' # Brownian Motion discretion (one minute)
# Trading Days
nyse = mcal.get_calendar('NYSE')
cal = mcal.date_range(nyse.schedule(start_date = idate, end_date = vdate), frequency = fre)
m = len(list(cal))
dt = 1/m
r = con.ref('USGG12M Index', 'PX_LAST')['value'][0]/100 # Annual Risk-free rate (takes USD rate, should be amended to relevant FX)
dr = r/m
start_hist = (datetime.date.today() + datetime.timedelta(days = -2*365)).strftime('%Y%m%d') # Start date for historical calculation
end_hist = (datetime.date.today() + datetime.timedelta(days = -1)).strftime('%Y%m%d') # End date for historical calculation



# Spot Prices
spots = [con.ref(sec, 'PX_LAST')['value'][0] for sec in underlyings]
# Barrier Level
Bs = [barrier*K for K in Ks]
# Dividend Yields
ds = [0 if isnan(d) else d for d in [con.ref(sec, 'DIVIDEND_INDICATED_YIELD')['value'][0]/100 for sec in underlyings]]
# Conversion Ratios
CR = [SD/K for K in Ks]

#Standard Deviations & Correlation Matrix
px_hist = con.bdh(underlyings, 'PX_LAST', start_hist, end_hist)
ret_hist = np.diff(np.log(px_hist), axis = 0).T
corr_mtrx = np.corrcoef(ret_hist)
A = np.linalg.cholesky(corr_mtrx)
sig = [con.ref(sec, 'IVOL_MONEYNESS', [('IVOL_MONEYNESS_LEVEL', round(B/S*10,0)*10)])['value'][0]/100 for sec, B, S in zip(underlyings, Bs, spots)]


### Simulation Design

# Initiate List to fill
NPV = []

for n in range(0, N):
    # Matrices of Standard Normal Variables
    Z0 = [np.random.normal(loc = 0, scale = 1, size = (m)) for sec in underlyings]
    Z = [sum([a*z for a, z in zip([a for a in As],Z0)]) for As in [a for a in A]]
    
    # Initiate Matrices to fill
    S = [s*np.ones(m+1) for s in spots]
    
    # Simulate potential paths
    for i in range(0, m):
        for j in range(0, len(S)):
            S[j][i+1] = S[j][i] * exp((dr - ds[j] - .5 * sig[j]**2) * dt + sig[j] * sqrt(dt) * Z[j][i])
    
    # Check whether Knock-In Event occured and whether Worst Performance is negative
    if sum([min(s) < b for s, b in zip(S,Bs)]) > 0 and sum([s[i + 1]/k - 1 < 0 for s, k in zip(S, Ks)]) > 0:
        # Check which performed worse
        idx = [s[i + 1]/k - 1 for s, k in zip(S, Ks)].index(min([s[i + 1]/k - 1 for s, k in zip(S, Ks)]))
        fv = sum([SD*IA/360*(cf_date - compound_date).days/(1+r/360)**(cf_date - max(datetime.date.today(), idate)).days for cf_date, compound_date in zip(cfdates[1:len(cfdates)], cfdates[:len(cfdates)-1]) if (cf_date - max(datetime.date.today(), idate)).days >= 0] + [CR[idx]*S[idx][i + 1]/(1+r/360)**(cfdates[len(cfdates) - 1] - max(datetime.date.today(), idate)).days])
    else:
        fv = sum([SD*IA/360*(cf_date - compound_date).days/(1+r/360)**(cf_date - max(datetime.date.today(), idate)).days for cf_date, compound_date in zip(cfdates[1:len(cfdates)], cfdates[:len(cfdates)-1]) if (cf_date - max(datetime.date.today(), idate)).days >= 0] + [SD/(1+r/360)**(cfdates[len(cfdates) - 1] - max(datetime.date.today(), idate)).days])

    # Save & Print outcome    
    NPV.append(fv)
    print(round(mean(NPV), 2))










