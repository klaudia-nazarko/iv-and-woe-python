import pandas as pd
import numpy as np

df = pd.read_csv('data/telco_churn.csv')
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











class Feature(object):

    label = 'label'
    agg = {
        label: ['count', 'sum']
    }

    def __init__(self, df, feature):
        self.feature = feature
        self.df_lite = df[[feature, self.label]]
        #self.df_grouped = self.group_by_feature()
        #self.df_grouped_with_perc_share = self.calculate_perc_share()
        #self.df_grouped_with_woe = self.calculate_woe()
        self.df_grouped_with_iv, self.iv = self.calculate_iv()

    def group_by_feature(self):
        self.df_grouped = self.df_lite \
                            .groupby(self.feature) \
                            .agg(self.agg) \
                            .reset_index()
        self.df_grouped.columns = [self.feature, 'count', 'good']
        self.df_grouped['bad'] = self.df_grouped['count'] - self.df_grouped['good']
        return self.df_grouped

    def perc_share(self, group_name):
        return self.df_grouped[group_name] / self.df_grouped[group_name].sum()

    def calculate_perc_share(self):
        self.df_grouped_with_perc_share = self.group_by_feature()
        self.df_grouped_with_perc_share['perc_good'] = self.perc_share('good')
        self.df_grouped_with_perc_share['perc_bad'] = self.perc_share('bad')
        self.df_grouped_with_perc_share['perc_diff'] = self.df_grouped_with_perc_share['perc_good'] - self.df_grouped_with_perc_share['perc_bad']
        return self.df_grouped_with_perc_share

    def calculate_woe(self):
        self.df_grouped_with_woe = self.calculate_perc_share()
        self.df_grouped_with_woe['woe'] = np.log(self.df_grouped_with_woe['perc_good']/self.df_grouped_with_woe['perc_bad'])
        return self.df_grouped_with_woe

    def calculate_iv(self):
        self.df_grouped_with_iv = self.calculate_woe()
        self.df_grouped_with_iv['iv'] = self.df_grouped_with_iv['perc_diff'] * self.df_grouped_with_iv['woe']
        return self.df_grouped_with_iv, self.df_grouped_with_iv['iv'].sum()





