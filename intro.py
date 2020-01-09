import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/telco_churn.csv', na_values=[' '])
df['label'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

class AttributeRelevance():
    def bulk_iv(self, feats, iv):
        iv_dict = {}
        for f in feats:
            iv_df, iv_value = iv.calculate_iv(f)
            iv_dict[f.feature] = iv_value
        df = pd.DataFrame.from_dict(iv_dict, orient='index', columns=['iv'])
        return df

    def bulk_stats(self, feats, s):
        stats_dict = {}
        for f in feats:
            p_value, effect_size = s.calculate_chi(f)
            stats_dict[f.feature] = [p_value, effect_size]
        df = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['p-value', 'effect_size'])
        return df

    def analyze(self, feats, iv, s=None):
        df_iv = self.bulk_iv(feats, iv).sort_values(by='iv', ascending=False)
        if s is None:
            return df_iv
        else:
            df_stats = self.bulk_stats(feats, s)
            return df_iv.merge(df_stats, left_index=True, right_index=True)



class Analysis():
    def group_by_feature(self, feat):
        df = feat.df_lite \
                            .groupby('bin') \
                            .agg({'label': ['count', 'sum']}) \
                            .reset_index()
        df.columns = [feat.feature, 'count', 'good']
        df['bad'] = df['count'] - df['good']
        return df

class StatsSignificance(Analysis):
    def calculate_chi(self, feat):
        df = self.group_by_feature(feat)
        df_chi = np.array(df[['good', 'bad']])
        n = df['count'].sum()

        chi = stats.chi2_contingency(df_chi)
        cramers_v = np.sqrt(chi[0] / n)          # assume that k=2 (good, bad)
        return chi[1], cramers_v

class IV(Analysis):
    @staticmethod
    def __perc_share(df, group_name):
        return df[group_name] / df[group_name].sum()

    def __calculate_perc_share(self, feat):
        df = self.group_by_feature(feat)
        df['perc_good'] = self.__perc_share(df, 'good')
        df['perc_bad'] = self.__perc_share(df, 'bad')
        df['perc_diff'] = df['perc_good'] - df['perc_bad']
        return df

    def __calculate_woe(self, feat):
        df = self.__calculate_perc_share(feat)
        df['woe'] = np.log(df['perc_good']/df['perc_bad'])
        df['woe'] = df['woe'].replace([np.inf, -np.inf], np.nan).fillna(0)
        return df

    def calculate_iv(self, feat):
        df = self.__calculate_woe(feat)
        df['iv'] = df['perc_diff'] * df['woe']
        return df, df['iv'].sum()

    def draw_woe(self, feat):
        with pd.option_context('mode.chained_assignment', None):
            iv_df, iv_value = self.calculate_iv(feat)
        fig, ax = plt.subplots(figsize=(10,6))
        palette = sns.color_palette("Set2")
        sns.barplot(x=feat.feature, y='woe', data=iv_df, palette=palette)
        ax.set_title('WOE visualization for: ' + feat.feature)
        plt.show()


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

    def __generate_bins(self, bins_num):
        df = self.df[[self.feature, 'label']]
        df['bin'] = pd.qcut(df[self.feature], bins_num, duplicates='drop') \
                    .apply(lambda x: x.left) \
                    .astype(float)
        return df

    def __generate_correct_bins(self, bins_max=20):
        for bins_num in range(bins_max, 1, -1):
            with pd.option_context('mode.chained_assignment', None):
                df = self.__generate_bins(bins_num)
            df_grouped = pd.DataFrame(df.groupby('bin') \
                                      .agg({self.feature: 'count',
                                            'label': 'sum'})) \
                                      .reset_index()
            r, p = stats.stats.spearmanr(df_grouped['bin'], df_grouped['label'])

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
            df_lite = self.__generate_correct_bins()
        df_lite['bin'].fillna('MISSING', inplace=True)
        return df_lite[['bin', 'label']]


feat_gender = CategoricalFeature(df, 'gender')
feat_charges = ContinuousFeature(df, 'TotalCharges')

iv = IV()
iv.calculate_iv(feat_gender)
iv.calculate_iv(feat_charges)

iv.draw_woe(feat_gender)
iv.draw_woe(feat_charges)

feat_contract = CategoricalFeature(df, 'Contract')
feat_tenure = ContinuousFeature(df, 'tenure')

s = StatsSignificance()

feats = [feat_gender, feat_charges, feat_contract, feat_tenure]
iva = AttributeRelevance()
iva.bulk_iv(feats, iv)

iva.bulk_stats(feats, s)

iva.analyze(feats, iv, s)

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