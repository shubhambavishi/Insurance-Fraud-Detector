import numpy as np
import pandas as pd
import os
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn import metrics
import pickle

insurance_data = pd.read_csv('./insurance_claims.csv')



# dropping unneccessary columns from the dataset
insurance_data.drop(['collision_type', 'policy_number', 'policy_bind_date', 'insured_zip', 'incident_date', 'incident_location', 'auto_model'], axis = 1, inplace = True)

replace_values = {'?' : 'UNKNOWN'}  
insurance_data = insurance_data.replace({"property_damage": replace_values, 'police_report_available': replace_values})



# Removing negative values from the columns
replace_values = {-1000000:0}
insurance_data = insurance_data.replace({"umbrella_limit": replace_values})
insurance_data['capital-loss'] = insurance_data['capital-loss'].abs()

# Here compare only numerical features
corr_mat = insurance_data.corr().abs()
high_corr_var=np.where(corr_mat>0.8)

# removing the highly correlated numerical column 
insurance_data.drop(['age', 'injury_claim', 'property_claim', 'vehicle_claim'], axis=1, inplace=True)

# Here we only compare catagorical features

alpha = 0.05
for cols_1 in insurance_data.columns:
    if insurance_data[cols_1].dtype == 'object':
        for cols_2 in insurance_data.columns:
            if insurance_data[cols_2].dtype == 'object':
                table = pd.crosstab(insurance_data[cols_1], insurance_data[cols_2], margins = False)     
                stat, p, dof, expected = chi2_contingency(table)
                if p <= alpha:
                    #'Variables are associated (reject H0)'
                    if cols_1 != cols_2:
                    	continue
                        # note down the columns
                else:
                    continue
                    #'Variables are not associated(fail to reject H0)'

# removing columns which gave p-value less than alpha
insurance_data.drop(['incident_state', 'police_report_available', 'incident_severity', 'authorities_contacted', 'property_damage'], axis = 1, inplace = True)

# insurance_data.to_csv('./cleaned_data.csv')


for cols in insurance_data.columns:
    if insurance_data[cols].dtypes == 'object':
        if cols != 'fraud_reported':
            one_hot = pd.get_dummies(insurance_data[cols])
            insurance_data.drop(cols,axis = 1, inplace = True)
            insurance_data = pd.concat([insurance_data, one_hot], axis=1)

insurance_data = insurance_data.append(insurance_data.loc[insurance_data['fraud_reported'] == 'Y'])

y = insurance_data['fraud_reported']
X = insurance_data.drop('fraud_reported', axis = 1)

y = pd.get_dummies(y, drop_first=True)



kfold = StratifiedKFold(n_splits=5)
kfold.get_n_splits(X, y)

for train_index, test_index in kfold.split(X, y):
    X_train = X.iloc[train_index,:]
    X_test = X.iloc[test_index,:]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:,1]
    
    matrix = confusion_matrix(np.where(y_pred >= 0.6, 1, 0), y_test.iloc[:])
    fpr = matrix[0,1]/(matrix[0,1]+matrix[1,1])
    tpr = matrix[0,0]/(matrix[0,0]+matrix[1,0])



filename = './rfc_model.sav'
pickle.dump(model, open(filename, 'wb'))


loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model.predict(X.iloc[100:101,:]))