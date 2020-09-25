# Insurance-Fraud-Detector

Overview: This detection is made for the people who are in dilemma whether to invest in car insurance of not. This project is totally based on Machine Learning.
If you want to directly visit the flask application. Here is the link: https://insurane-fraud.herokuapp.com

Project pipleline:
### Data Gathering:
I found out this dataset from kaggle https://www.kaggle.com/buntyshah/auto-insurance-claims-data , which has 1000 rows with nearly 40 columns.
### Data Preprocessing:
First i removed all the columns which we can identify unneccessary, like date of policy,id, area pincode.
It was easy and fast preprocessing because of not null values, but there are some values which were unknown denoted as ? in dataset. I replaced it with UNKNOWN string.
### Feature Selection:
Using pearson co-efficient i removed columns which have more than 80% of correlation. Also for categorical values, i had performed Chi-square testing to reject the highly correlated columns.
After feature selection, only 23 columns were accounted.
### Data Visualization:
I have uploaded images which includes histogram, scatter plot and box plot.
When i saw scatter plot of total_claim_amount and fraud_reported, an intresting pattern i found out. We can say based on image that of you have lower claim amount, there are less chances to get fraud.
### Model Creation:
First of all, i have encountered imbalanced dataset.So, for train test split , StratifiedKFold is proven important. Also after appling KNN and Random Forst Regression , i found out that output was totally biased toward FP. So, to handle this issue i applied upsampling in training dataset.
### Model Evaluation:
As we know that this dataset has more values of N compared to Y, just computing accuracy using confusion matrics is not worthy. So, here i coded manually Roc-Auc curve. We required higher TPR and much lower FPR. So, with going to threshold = 0.6, i acquire FPR = 0.03 TPR=0.96 on testing dataset.
### Model Deployment:
Now for creating web application, flask is the first choice over any other framework as it is easy to code. And lastly, deployment of the model was done in Heroku platform.
