import pandas as pd
import numpy as np
import scipy.stats.stats as stats

df = pd.read_csv('data/telco_churn.csv', na_values=[' '])
df['label'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

class IV():
    agg = {
        'label': ['count', 'sum']
    }

    def group_by_feature(self, feat):
        df = feat.df_lite \
                            .groupby('bin') \
                            .agg(self.agg) \
                            .reset_index()
        df.columns = [feat.feature, 'count', 'good']
        df['bad'] = df['count'] - df['good']
        return df

    @staticmethod
    def perc_share(df, group_name):
        return df[group_name] / df[group_name].sum()

    def calculate_perc_share(self, feat):
        df = self.group_by_feature(feat)
        df['perc_good'] = self.perc_share(df, 'good')
        df['perc_bad'] = self.perc_share(df, 'bad')
        df['perc_diff'] = df['perc_good'] - df['perc_bad']
        return df

    def calculate_woe(self, feat):
        df = self.calculate_perc_share(feat)
        df['woe'] = np.log(df['perc_good']/df['perc_bad'])
        df['woe'] = df['woe'].replace([np.inf, -np.inf], np.nan).fillna(0)
        return df

    def calculate_iv(self, feat):
        df = self.calculate_woe(feat)
        df['iv'] = df['perc_diff'] * df['woe']
        return df, df['iv'].sum()


class CategoricalFeature():
    def __init__(self, df, feature):
        self.df = df
        self.feature = feature

    @property
    def df_lite(self):
        with pd.option_context('mode.chained_assignment', None):
            df_lite = self.df
        df_lite['bin'] = df_lite[self.feature].fillna('MISSING')
        return df_lite[['bin', 'label']]


class ContinuousFeature():
    def __init__(self, df, feature):
        self.df = df
        self.feature = feature
        self.bin_min_size = int(len(self.df) * 0.05)

    def generate_bins(self, bins_num):
        df = self.df[[self.feature, 'label']]
        df['bin'] = pd.qcut(df[self.feature], bins_num, duplicates='drop').apply(lambda x: x.left).astype(float)
        return df

    def generate_correct_bins(self, bins_max=20):
        for bins_num in range(bins_max, 1, -1):
            with pd.option_context('mode.chained_assignment', None):
                df = self.generate_bins(bins_num)
            df_grouped = pd.DataFrame(df.groupby('bin') \
                                      .agg({self.feature: 'count',
                                            'label': 'sum'})) \
                                      .reset_index()
            r, p = stats.spearmanr(df_grouped['bin'], df_grouped['label'])

            if (
                    abs(r)==1 and                                                        # check if woe for bins are monotonic
                    df_grouped[self.feature].min() > self.bin_min_size                   # check if bin size is greater than 5%
                    and not (df_grouped[self.feature] == df_grouped['label']).any()      # check if number of good and bad is not equal to 0
            ):
                break

        return df

    @property
    def df_lite(self):
        with pd.option_context('mode.chained_assignment', None):
            df_lite = self.generate_correct_bins()
        df_lite['bin'].fillna('MISSING', inplace=True)
        return df_lite[['bin', 'label']]


feat_gender = CategoricalFeature(df, 'gender')
feat_charges = ContinuousFeature(df, 'TotalCharges')

iv = IV()
iv.calculate_iv(feat_gender)
iv.calculate_iv(feat_charges)

#feat_contract = CategoricalFeature(df, 'Contract')
#feat_tenure = ContinuousFeature(df, 'tenure')


### Continuous variables
# Group NaNs together
# Split into X (eg 20 / 10) equal-size bins
# Check if each bin is bigger than 5% of all observations
# Non zero for events and non-events
# Check if woe is monotonic
# Iterate




charges = ContinuousFeature(df, 'TotalCharges')
x = charges.generate_correct_bins(20)

tenure = ContinuousFeature(df, 'tenure')
x = tenure.generate_correct_bins(20)


########

feature = 'TotalCharges'
bins_num = 20
df[feature] = pd.to_numeric(df[feature])

while bins_num > 0:
    df['bin'] = pd.qcut(df[feature], bins_num, labels=False, duplicates='drop')
    check = pd.DataFrame(df.groupby('bin').agg({feature: 'count', 'label': 'sum'})).reset_index()
    r, p = stats.spearmanr(check['bin'], check['label'])
    print(bins_num, r)
    bins_num -= 1