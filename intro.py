import pandas as pd
import numpy as np

df = pd.read_csv('data/telco_churn.csv', na_values=[' '])
df['label'] = df['Churn'].map({'Yes': 1, 'No': 0})


class Feature(object):

    label = 'label'
    agg = {
        label: ['count', 'sum']
    }

    def __init__(self, df, feature):
        self.feature = feature
        self.df_lite = df[[feature, self.label]]
        self.df_with_iv, self.iv = None, None

    def group_by_feature(self):
        df = self.df_lite \
                            .groupby(self.feature) \
                            .agg(self.agg) \
                            .reset_index()
        df.columns = [self.feature, 'count', 'good']
        df['bad'] = df['count'] - df['good']
        return df

    @staticmethod
    def perc_share(df, group_name):
        return df[group_name] / df[group_name].sum()

    def calculate_perc_share(self):
        df = self.group_by_feature()
        df['perc_good'] = self.perc_share(df, 'good')
        df['perc_bad'] = self.perc_share(df, 'bad')
        df['perc_diff'] = df['perc_good'] - df['perc_bad']
        return df

    def calculate_woe(self):
        df = self.calculate_perc_share()
        df['woe'] = np.log(df['perc_good']/df['perc_bad'])
        return df

    def calculate_iv(self):
        df = self.calculate_woe()
        df['iv'] = df['perc_diff'] * df['woe']
        self.df_with_iv, self.iv = df, df['iv'].sum()
        return df, df['iv'].sum()


feat_gender = Feature(df, 'gender')
feat_contract = Feature(df, 'Contract')


### Continuous variables
# Group NaNs together
# Split into X (eg 20 / 10) equal-size bins
# Check if each bin is bigger than 5% of all observations
# Non zero for events and non-events
# Check if woe is monotonic
# Iterate

import scipy.stats.stats as stats

class ContinuousFeature(Feature):
    #label = 'label'

    def __init__(self, df, feature):
        super().__init__(df, feature)
        self.bin_min_size = int(len(self.df_lite) * 0.05)

    def generate_bins(self, bins_num):
        df = self.df_lite
        df['bin'] = pd.qcut(df[self.feature], bins_num, labels=False, duplicates='drop')
        return df

    def generate_correct_bins(self, bins_max):
        for bins_num in range(bins_max, 1, -1):
            with pd.option_context('mode.chained_assignment', None):
                df = self.generate_bins(bins_num)
            df_grouped = pd.DataFrame(df.groupby('bin') \
                                      .agg({self.feature: 'count',
                                            self.label: 'sum'})) \
                                      .reset_index()
            r, p = stats.spearmanr(df_grouped['bin'], df_grouped[self.label])

            if (
                    abs(r)==1 and       # check if woe for bins are monotonic
                    df_grouped[self.feature].min() > self.bin_min_size      # check if bin size is greater than 5%
                    and not (df_grouped[self.feature] == df_grouped[self.label]).any()      # check if number of good and bad is not equal to 0
            ):
                break

        return df


charges = ContinuousFeature(df, 'TotalCharges')
x = charges.generate_correct_bins(20)

tenure = ContinuousFeature(df, 'tenure')
x = tenure.generate_correct_bins(20)


########

feature = 'tenure'
bins_num = 20
df[feature] = pd.to_numeric(df[feature])

while bins_num > 0:
    df['bin'] = pd.qcut(df[feature], bins_num, labels=False, duplicates='drop')
    check = pd.DataFrame(df.groupby('bin').agg({feature: 'count', 'label': 'sum'})).reset_index()
    r, p = stats.spearmanr(check['bin'], check['label'])
    print(bins_num, r)
    bins_num -= 1