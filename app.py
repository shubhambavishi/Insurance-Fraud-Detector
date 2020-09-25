import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('./rfc_model.sav', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	features = []
	features.append(int(request.form['month_as_customer']))
	features.append(request.form['policy_state'])
	features.append(request.form['policy_csl'])
	features.append(int(request.form['policy_deductable']))
	features.append(int(request.form['policy_annual_premium']))
	features.append(int(request.form['umbrella_limit']))
	features.append(request.form['insured_sex'])
	features.append(request.form['insured_education_level'])
	features.append(request.form['insured_occupation'])
	features.append(request.form['insured_hobbies'])
	features.append(request.form['insured_relationship'])
	features.append(int(request.form['capital-gains']))
	features.append(int(request.form['capital-loss']))
	features.append(request.form['incident_type'])
	features.append(request.form['incident_city'])
	features.append(int(request.form['incident_hour_of_the_day']))
	features.append(int(request.form['number_of_vehicles_involved']))
	features.append(int(request.form['bodily_injuries']))    
	features.append(int(request.form['witnesses']))
	features.append(int(request.form['total_claim_amount']))
	features.append(request.form['auto_make'])
	features.append(int(request.form['auto_year']))
	cols = ['months_as_customer', 'policy_state', 'policy_csl', 'policy_deductable', 'policy_annual_premium', 'umbrella_limit', 'insured_sex', 'insured_education_level', 'insured_occupation', 'insured_hobbies', 'insured_relationship', 'capital-gains', 'capital-loss', 'incident_type', 'incident_city', 'incident_hour_of_the_day', 'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'total_claim_amount', 'auto_make', 'auto_year']
	data = {}
	insurance_data = pd.read_csv('./cleaned_data.csv')
	insurance_data.drop(['index', 'fraud_reported'], axis=1, inplace=True)
	for i in range(len(cols)):
		data.update({cols[i] : features[i]})

	print (data)
	insurance_data = insurance_data.append(data, ignore_index=True)

	for cols in insurance_data.columns:
		if insurance_data[cols].dtypes == 'object':
			one_hot = pd.get_dummies(insurance_data[cols])
			insurance_data.drop(cols, axis=1, inplace = True)
			insurance_data = pd.concat([insurance_data, one_hot], axis=1)

	pred_data = insurance_data.iloc[-1, :]
	y_pred = model.predict_proba(np.reshape(pred_data, (1,92)))[:,1]
	threshold = 0.6
	if y_pred > threshold:
		return render_template('index.html', prediction_text='Alert!!! You need to re-check security')
	else:
		return render_template('index.html', prediction_text='You are safe')

if __name__ == "__main__":
    app.run(debug=True)

