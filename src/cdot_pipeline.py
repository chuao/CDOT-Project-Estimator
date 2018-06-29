
import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from cpi import to_2017
pd.options.display.float_format='{:.4g}'.format
RANDOM_SEED = 1234567890


class pipeline():
    '''
    This class loads the data files and creates a preprocessed dataframe
    with the features needed to perform a linear regression
    '''
    def __init__(self):
        pass


    def cleandf(self):
        '''
        Loads the relevant datafiles and preprocess the data
        combining two dataframes, filtering, slicing and
        aggregating the data into a dataframe suitable for
        Regression, with clear column names.
        '''
        # First we load the relevant datafiles
        bid = pd.read_csv('../data/bidding_info.csv', low_memory=False)
        change_orders = pd.read_csv('../data/change_orders.csv',
                                    low_memory=False)

        # Drop the records wirh Awarded column == NaN from bid
        bid = bid[bid['Awarded'].notnull()]
        # And the records with the Bid Total NaN
        bid = bid[bid['Bid Total'].notnull()]

        # create a list of sucessful bids (someone won)
        # otherwise there is no actual
        #
        # and drop unsuccesful bids from tables
        successful_bids = bid[bid['Awarded'] == 1]['Proposal Number']
        bid = bid[bid['Proposal Number'].isin(successful_bids)]
        change_orders = change_orders[change_orders['CONT_ID'].isin(successful_bids)]

        # Convert dollar amounts from strings to float
        # first for bid
        columns_to_convert = ['Bid Total', 'Engineers Estimate', 'Difference']
        for col in columns_to_convert:
            tmp = bid[col]
            bid[col] = tmp.map(self.dollar_val, na_action='ignore')
        # Next for change_orders
        columns_to_convert = ['C_O_AMT']
        for col in columns_to_convert:
            tmp = change_orders[col]
            change_orders[col] = tmp.map(self.dollar_val, na_action='ignore')

        # We add date columns to adjust for inflation
        bid['month'] = pd.to_datetime(bid['Letting Date']).dt.month
        bid['year'] = pd.to_datetime(bid['Letting Date']).dt.year
        bid['date'] = bid.apply(lambda x:pd.Timestamp.strptime(
                               "{0} {1}".format(x['year'],x['month']),
                                "%Y %m"),axis=1)
        change_orders['date'] = change_orders.apply(lambda
                                x:pd.Timestamp.strptime("{0}".format(x['APPR_DT']//100),
                                 "%Y%m"),axis=1)

        # Convert all the money columns to 2017 dollars
        # so we can compae apples to apples
        bid['Bid Total 2017'] = np.vectorize(to_2017)(bid['Bid Total'], bid['date'].astype(str))
        bid['Engineers Estimate 2017'] = np.vectorize(to_2017)(bid['Engineers Estimate'], bid['date'].astype(str))
        change_orders['C_O_AM_2017'] = np.vectorize(to_2017)(change_orders['C_O_AMT'], change_orders['date'].astype(str))

        # Dropping unnecessary columns
        cols_to_drop = ['Vendor Number', 'Vendor Name', 'Difference','Bid Total',
                        'Engineers Estimate', 'Percentage Bid Total Over Estimate',
                        'Letting Date','month', 'year']
        bid = bid.drop(cols_to_drop, axis=1)
        cols_to_drop = ['Vendor', 'Bid Amount', 'LEV2_OFFICE_NBR', 'DESC1','C_O_AMT',
                        'C_O_NBR', 'CD_DESC', 'APPR_DT', 'CHNG_DESC', 'C_O_T']
        change_orders = change_orders.drop(cols_to_drop, axis=1)

        # Aggregate changes per contract
        ch_o_total = change_orders.groupby('CONT_ID').sum()
        ch_o_total.reset_index(inplace=True)
        ch_o_total.columns = ['Proposal Number', 'C_O_AM_2017']

        # Merge with bid
        bid = pd.merge(bid,ch_o_total, on='Proposal Number', how='outer' ).sort_values(by=['Proposal Number', 'Awarded'])
        # no reported changes is assumed as the bid total was correct and is the actual expense
        bid.fillna(0, inplace=True)
        bid['Final cost 2017'] = bid['Bid Total 2017'] + bid['C_O_AM_2017']
        self.bid = bid
        return 'DataFrame created'

    def dollar_val(self, strng, mylocale='en_US.UTF-8'):
        '''Function to convert formatted string in database to float of dollar amounts
        Generalized to any locale for possible reutilization in the future
        '''
        import locale
        locale.setlocale(locale.LC_ALL, mylocale)
        locale._override_localeconv = {'n_sign_posn':0}
        conv = locale.localeconv()
        if type(strng) == str:
            raw_numbers = strng.replace(conv['currency_symbol'], '')
        else:
            raw_numbers = strng
        return locale.atof(raw_numbers)


    def make_feat(self):
        '''
        Function to compute some statistics about winners and losers
        of the bids and use them to creat features like Eccentricity
        or mean of bids


        INPUT: dataframe with ['Proposal Number', 'Awarded', 'date',
                               'Bid Total 2017','Engineers Estimate 2017',
                               'C_O_AM_2017']
        OUTPUT: Feature and Target DataFrame
        '''
        bid = self.bid
        g_loser = bid[bid['Awarded'] == 0].groupby('Proposal Number')
        bid = bid[bid['Awarded'] == 1]

        # Make aggregated dataframe
        loser = pd.DataFrame(g_loser.agg(['size', 'mean', 'min', 'max']))
        # flatten column indexes, honestly, because I don't know how to use them
        loser.columns = [' '.join(col) for col in loser.columns]
        loser.reset_index(inplace=True)

        # Dropping unnecessary columns
        cols_to_drop = ['Awarded mean', 'Awarded min', 'Awarded max',
                        'Bid Total 2017 size', 'Engineers Estimate 2017 size',
                        'Engineers Estimate 2017 mean', 'Engineers Estimate 2017 min',
                        'Engineers Estimate 2017 max', 'C_O_AM_2017 size',
                        'C_O_AM_2017 mean', 'C_O_AM_2017 min',
                        'C_O_AM_2017 max', 'Final cost 2017 size',
                        'Final cost 2017 mean', 'Final cost 2017 min',
                        'Final cost 2017 max']
        loser = loser.drop(cols_to_drop, axis=1)
        loser.columns = ['Proposal Number', 'No of participants',
                         'Loser Bid 2017 mean', 'Loser Bid 2017 min',
                         'Loser Bid 2017 max']
        bid = pd.merge(bid,loser, on='Proposal Number', how='outer' )
        bid.fillna(0, inplace=True)
        bid['No of participants'] += 1
        bid = bid.drop('Awarded', axis=1)

# Add winner bid to average and set mean to winner when only one bidder
        bid['Loser Bid 2017 mean'] = np.where((bid['No of participants'] == 1),
                                         bid['Bid Total 2017'],
                                        (bid['Loser Bid 2017 mean'] +
                                        (bid['No of participants'] - 1) *
                                         bid['Bid Total 2017']) /
                                         bid['No of participants'])

# Add winner bid to min and set min to winner when only one bidder
        bid['Loser Bid 2017 min'] = np.where((bid['No of participants'] == 1),
                                        bid['Bid Total 2017'],
                                        np.minimum(bid['Loser Bid 2017 min'],
                                                   bid['Bid Total 2017']))

# Add winner bid to max and set max to winner when only one bidder
        bid['Loser Bid 2017 max'] = np.where((bid['No of participants'] == 1),
                                        bid['Bid Total 2017'],
                                        np.maximum(bid['Loser Bid 2017 max'],
                                                   bid['Bid Total 2017']))

# Rename columns to more meaninful and correct names
        bid.columns = ['Proposal Number', 'date', 'Winning Bid 2017', 'Engineers Estimate 2017',
                       'Changes 2017', 'Final cost 2017', 'No of participants', 'Bid 2017 mean',
                       'Bid 2017 min', 'Bid 2017 max']

# Create features Spread, and Eccentricity  of the winner and of the mean (1 and 2)
        bid['Spread'] = bid['Bid 2017 max'] - bid['Bid 2017 min']
        bid['Eccentricity 1'] = 0
        bid['Eccentricity 1'] = np.where((bid['Spread'] == 0),
                                        0,
                                        (bid['Bid 2017 min'] +
                                         bid['Bid 2017 max'] -
                                         2 * bid['Winning Bid 2017']) /
                                         bid['Spread'])
        bid['Eccentricity 2'] = 0
        bid['Eccentricity 2'] = np.where((bid['Spread'] == 0),
                                        0,
                                        (bid['Bid 2017 min'] +
                                         bid['Bid 2017 max'] -
                                         2 * bid['Bid 2017 mean']) /
                                         bid['Spread'])
        self.bid = bid
        return 'OK'








if __name__ == '__main__':
    p = pipeline()
    p.cleandf()
    print(p.bid)
    p.make_feat()
    print(p.bid)
