import pandas as pd

df = pd.read_csv('data/telco_churn.csv')
df['label'] = df['Churn'].map({'Yes': 1, 'No': 0})


class Feature(object):

    label = 'label'

    def __init__(self, df, feature):
        self.feature = feature
        self.df_lite = df[[feature, self.label]]
        self.df_grouped = self.group_by_feature()

    def group_by_feature(self):
        self.df_grouped = self.df_lite \
                            .groupby(self.feature) \
                            [self.label].count() \
                            .reset_index() \
                            .rename({self.label: 'count'}, axis=1)
        return self.df_grouped

feat = Feature(df, 'gender')
feat.df_grouped