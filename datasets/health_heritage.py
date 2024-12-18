"""
The preprocessing is taken from https://github.com/eth-sri/fnf (Apache License: https://github.com/eth-sri/fnf/blob/main/LICENSE),
who adapted it from https://github.com/truongkhanhduy95/Heritage-Health-Prize.
"""
'''
Refence: https://github.com/eth-sri/tableak
'''
import torch
import numpy as np
import pandas as pd
import sys

from my_utils.utils import to_numeric, to_categorical
from sklearn.model_selection import train_test_split
from my_utils.utils import tensor_data_create

class HealthHeritage():

    def __init__(self, property='sex'):
        # self.train_test_ratio = train_test_ratio
        # self.random_state = random_state
        # self.standardized = False

        prop_list = {
            'sex': 'Sex=F', 
            'age': 'AgeAtFirstClaim=80+' 
        }

        self.features = {
            'LabCount_total': None,
            'LabCount_months': None,
            'DrugCount_total': None,
            'DrugCount_months': None,
            'no_Claims': None,
            'no_Providers': None,
            'no_Vendors': None,
            'no_PCPs': None,
            'max_CharlsonIndex': None,
            'PayDelay_total': None,
            'PayDelay_max': None,
            'PayDelay_min': None,
            'PrimaryConditionGroup': ['AMI', 'APPCHOL', 'ARTHSPIN', 'CANCRA', 'CANCRB', 'CANCRM', 'CATAST', 'CHF',
                                      'COPD', 'FLaELEC', 'FXDISLC', 'GIBLEED',
                                      'GIOBSENT', 'GYNEC1', 'GYNECA', 'HEART2', 'HEART4', 'HEMTOL', 'HIPFX', 'INFEC4',
                                      'LIVERDZ', 'METAB1', 'METAB3',
                                      'MISCHRT', 'MISCL1', 'MISCL5', 'MSC2a3', 'NEUMENT', 'ODaBNCA', 'PERINTL',
                                      'PERVALV', 'PNCRDZ', 'PNEUM', 'PRGNCY',
                                      'PrimaryConditionGroup_?', 'RENAL1', 'RENAL2', 'RENAL3', 'RESPR4', 'ROAMI',
                                      'SEIZURE', 'SEPSIS', 'SKNAUT', 'STROKE',
                                      'TRAUMA', 'UTI'],
            'Specialty': ['Anesthesiology', 'Diagnostic Imaging', 'Emergency', 'General Practice', 'Internal',
                          'Laboratory', 'Obstetrics and Gynecology',
                          'Other', 'Pathology', 'Pediatrics', 'Rehabilitation', 'Specialty_?', 'Surgery'],
            'ProcedureGroup': ['ANES', 'EM', 'MED', 'PL', 'ProcedureGroup_?', 'RAD', 'SAS', 'SCS', 'SDS', 'SEOA', 'SGS',
                               'SIS', 'SMCD', 'SMS', 'SNS',
                               'SO', 'SRS', 'SUS'],
            'PlaceSvc': ['Ambulance', 'Home', 'Independent Lab', 'Inpatient Hospital', 'Office', 'Other',
                         'Outpatient Hospital', 'PlaceSvc_?', 'Urgent Care'],
            'AgeAtFirstClaim': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+', '?'],
            'Sex': ['?', 'F', 'M']
        }

        self.label = 'max_CharlsonIndex'

        df_claims = self.preprocess_claims(pd.read_csv('../data/Health_Heritage/Claims1.csv', sep=','),
                                      ['PrimaryConditionGroup', 'Specialty', 'ProcedureGroup', 'PlaceSvc'])
        df_drugs = self.preprocess_drugs(pd.read_csv('../data/Health_Heritage/DrugCount1.csv', sep=','))
        df_labs = self.preprocess_labs(pd.read_csv('../data/Health_Heritage/LabCount1.csv', sep=','))
        df_members = self.preprocess_members(pd.read_csv('../data/Health_Heritage/Members1.csv', sep=','))

        df_labs_drugs = pd.merge(df_labs, df_drugs, on=['MemberID', 'Year'], how='outer')
        df_labs_drugs_claims = pd.merge(df_labs_drugs, df_claims, on=['MemberID', 'Year'], how='outer')
        df_health = pd.merge(df_labs_drugs_claims, df_members, on=['MemberID'], how='outer')

        df_health.drop(['Year', 'MemberID'], axis=1, inplace=True)
        df_health.fillna(0, inplace=True)

        prop_key = prop_list[property]
        prop = df_health[prop_key]

        columns = list(df_health.columns)
        columns.remove(prop_key)  
        new_order = [prop_key] + columns  
        df_health = df_health[new_order]

        health_np = df_health.to_numpy()
        health_np_mixed = to_categorical(health_np, self.features)

        # create the labels
        for i, feature in enumerate(self.features.keys()):
            if feature == 'max_CharlsonIndex':
                for j in range(len(health_np_mixed[:, i])):
                    health_np_mixed[j, i] = '>0' if (health_np_mixed[j, i].astype(np.float32) > 0) else '=0'


        self.features['max_CharlsonIndex'] = ['=0', '>0']
        self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}

        # convert back to now already the correct numeric representation
        health_np = to_numeric(health_np_mixed, self.features, label=self.label).astype(np.float32)

        # split the labels and the features
        non_label_indices = [i for i in range(health_np.shape[1]) if i != 8]
        X, y = health_np[:, non_label_indices], health_np[:, 8]
        self.num_features = X.shape[1]

        # reorder the features directory to put the labels at the back
        del self.features['max_CharlsonIndex']
        self.features['max_CharlsonIndex'] = ['=0', '>0']

        # split the labels and the features
        non_label_indices = [i for i in range(health_np.shape[1]) if i != 8]
        x_data, y_data = health_np[:, non_label_indices], health_np[:, 8]

        self.prop = prop.to_numpy()
        self.data = tensor_data_create(x_data, y_data) #(61700, 110) (61700,)

        print("Percent of positive classes: {:.2%}".format(np.mean(y_data)))
        print("Percent of target property {}={}: {:.2%}".format(property, prop_list[property], np.mean(self.prop)))

    
    def get_data(self):
        return list(self.data), (self.prop)


    @staticmethod
    def preprocess_claims(df_claims, cat_names):
        df_claims.loc[df_claims['PayDelay'] == '162+', 'PayDelay'] = 162
        df_claims['PayDelay'] = df_claims['PayDelay'].astype(int)

        df_claims.loc[df_claims['DSFS'] == '0- 1 month', 'DSFS'] = 1
        df_claims.loc[df_claims['DSFS'] == '1- 2 months', 'DSFS'] = 2
        df_claims.loc[df_claims['DSFS'] == '2- 3 months', 'DSFS'] = 3
        df_claims.loc[df_claims['DSFS'] == '3- 4 months', 'DSFS'] = 4
        df_claims.loc[df_claims['DSFS'] == '4- 5 months', 'DSFS'] = 5
        df_claims.loc[df_claims['DSFS'] == '5- 6 months', 'DSFS'] = 6
        df_claims.loc[df_claims['DSFS'] == '6- 7 months', 'DSFS'] = 7
        df_claims.loc[df_claims['DSFS'] == '7- 8 months', 'DSFS'] = 8
        df_claims.loc[df_claims['DSFS'] == '8- 9 months', 'DSFS'] = 9
        df_claims.loc[df_claims['DSFS'] == '9-10 months', 'DSFS'] = 10
        df_claims.loc[df_claims['DSFS'] == '10-11 months', 'DSFS'] = 11
        df_claims.loc[df_claims['DSFS'] == '11-12 months', 'DSFS'] = 12

        df_claims.loc[df_claims['CharlsonIndex'] == '0', 'CharlsonIndex'] = 0
        df_claims.loc[df_claims['CharlsonIndex'] == '1-2', 'CharlsonIndex'] = 1
        df_claims.loc[df_claims['CharlsonIndex'] == '3-4', 'CharlsonIndex'] = 2
        df_claims.loc[df_claims['CharlsonIndex'] == '5+', 'CharlsonIndex'] = 3

        df_claims.loc[df_claims['LengthOfStay'] == '1 day', 'LengthOfStay'] = 1
        df_claims.loc[df_claims['LengthOfStay'] == '2 days', 'LengthOfStay'] = 2
        df_claims.loc[df_claims['LengthOfStay'] == '3 days', 'LengthOfStay'] = 3
        df_claims.loc[df_claims['LengthOfStay'] == '4 days', 'LengthOfStay'] = 4
        df_claims.loc[df_claims['LengthOfStay'] == '5 days', 'LengthOfStay'] = 5
        df_claims.loc[df_claims['LengthOfStay'] == '6 days', 'LengthOfStay'] = 6
        df_claims.loc[df_claims['LengthOfStay'] == '1- 2 weeks', 'LengthOfStay'] = 11
        df_claims.loc[df_claims['LengthOfStay'] == '2- 4 weeks', 'LengthOfStay'] = 21
        df_claims.loc[df_claims['LengthOfStay'] == '4- 8 weeks', 'LengthOfStay'] = 42
        df_claims.loc[df_claims['LengthOfStay'] == '26+ weeks', 'LengthOfStay'] = 180
        df_claims['LengthOfStay'].fillna(0, inplace=True)
        df_claims['LengthOfStay'] = df_claims['LengthOfStay'].astype(int)

        for cat_name in cat_names:
            df_claims[cat_name].fillna(f'{cat_name}_?', inplace=True)
        df_claims = pd.get_dummies(df_claims, columns=cat_names, prefix_sep='=')

        oh = [col for col in df_claims if '=' in col]

        agg = {
            'ProviderID': ['count', 'nunique'],
            'Vendor': 'nunique',
            'PCP': 'nunique',
            'CharlsonIndex': 'max',
            # 'PlaceSvc': 'nunique',
            # 'Specialty': 'nunique',
            # 'PrimaryConditionGroup': 'nunique',
            # 'ProcedureGroup': 'nunique',
            'PayDelay': ['sum', 'max', 'min']
        }
        for col in oh:
            agg[col] = 'sum'

        df_group = df_claims.groupby(['Year', 'MemberID'])
        df_claims = df_group.agg(agg).reset_index()
        df_claims.columns = [
                                'Year', 'MemberID', 'no_Claims', 'no_Providers', 'no_Vendors', 'no_PCPs',
                                'max_CharlsonIndex', 'PayDelay_total', 'PayDelay_max', 'PayDelay_min'
                            ] + oh

        return df_claims

    @staticmethod
    def preprocess_drugs(df_drugs):
        df_drugs.drop(columns=['DSFS'], inplace=True)
        # df_drugs['DSFS'] = df_drugs['DSFS'].apply(lambda x: int(x.split('-')[0])+1)
        df_drugs['DrugCount'] = df_drugs['DrugCount'].apply(lambda x: int(x.replace('+', '')))
        df_drugs = df_drugs.groupby(['Year', 'MemberID']).agg({'DrugCount': ['sum', 'count']}).reset_index()
        df_drugs.columns = ['Year', 'MemberID', 'DrugCount_total', 'DrugCount_months']
        # print('df_drugs.shape = ', df_drugs.shape)
        return df_drugs

    @staticmethod
    def preprocess_labs(df_labs):
        df_labs.drop(columns=['DSFS'], inplace=True)
        # df_labs['DSFS'] = df_labs['DSFS'].apply(lambda x: int(x.split('-')[0])+1)
        df_labs['LabCount'] = df_labs['LabCount'].apply(lambda x: int(x.replace('+', '')))
        df_labs = df_labs.groupby(['Year', 'MemberID']).agg({'LabCount': ['sum', 'count']}).reset_index()
        df_labs.columns = ['Year', 'MemberID', 'LabCount_total', 'LabCount_months']
        # print('df_labs.shape = ', df_labs.shape)
        return df_labs

    @staticmethod
    def preprocess_members(df_members):
        df_members['AgeAtFirstClaim'].fillna('?', inplace=True)
        df_members['Sex'].fillna('?', inplace=True)
        df_members = pd.get_dummies(
            df_members, columns=['AgeAtFirstClaim', 'Sex'], prefix_sep='='
        )
        # print('df_members.shape = ', df_members.shape)
        return df_members


# if __name__ == '__main__':
#     dataset = HealthHeritage('Sex').get_data()
#     dataset = HealthHeritage('AgeAtFirstClaim').get_data()

# Percent of positive classes: 29.82%
# Percent of target property Sex=F: 16.13%
# Percent of target property AgeAtFirstClaim=80+: 2.68%
