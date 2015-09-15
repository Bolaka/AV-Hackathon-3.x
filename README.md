# AV-Hackathon-3.x
> Predict customer worth for Happy Customer Bank with AUC of ~0.84

#### Data Preparation

**Lead_Creation_Date:** Read the dates as datetime objects in Python and transformed them to Gregorian ordinals

**DOB:** Extracted the year, month & day from the DOB column. Transformed DOB to Gregorian ordinals

**Gender:** Mapped as Female = 0, Male = 1

**Filled_Form:** Mapped as N = 0, Y = 1
    
**Device_Type:** Mapped as Mobile = 0, Web-browser = 1
    
**Mobile_Verified:** Mapped as N = 0, Y = 1

**City:** Ranked cities by counts and picked top 11. Labeled them numerically in _descending_ order. Rest of the cities labeled 0

**Employer_Name:** Cleaned outliers categories for 'TATA CONSULTANCY SERVICES LTD (TCS)'. Ranked employers by using the sum of Disbursed column and picked top 20. Labeled them numerically in _descending_ order (removing categories '0' & 'TYPE SLOWLY FOR AUTO FILL'). Rest of the employers labeled 0

**Salary_Account:** Ranked banks by counts and picked top 20. Labeled them numerically in _descending_ order. Rest of the banks labeled 0

**Var1:** Ranked var1 by counts and picked top 7. Labeled them numerically in _descending_ order. Rest of the var1 labeled 0

**Var2:** Ranked var2 by counts and picked all. Labeled them numerically in _descending_ order. Rest of the var2 labeled 0

**Source:** Ranked source by counts and picked top 7. Labeled them numerically in _descending_ order. Rest of the source labeled 0

#### Feature transformations

Looking at the histograms of Monthly_Income, Loan_Amount_Applied, Existing_EMI, Loan_Amount_Submitted, DOB_yr & Processing_Fee, skewness was detected. Following transformations were applied:

**Monthly_Income:** Square root of Monthly_Income replaced Monthly_Income

**Loan_Amount_Applied:** Square root of Loan_Amount_Applied replaced Loan_Amount_Applied

**Existing_EMI:** Cube root of Existing_EMI replaced Existing_EMI

**Loan_Amount_Submitted:** Square root of Loan_Amount_Submitted replaced Loan_Amount_Submitted

**DOB_yr:** Natural logarithm of DOB_yr replaced DOB_yr

**Processing_Fee:** Square root of Processing_Fee replaced Processing_Fee

#### Outliers

Replaced a value in columns **Monthly_Income, Loan_Amount_Applied, Existing_EMI & EMI_Loan_Submitted** by NaN if greater than 10 standard deviations away from the mean of that column.

#### Feature Selection

Kept all features except **DOB and Lead_Creation_Date**.

#### Cross validation

Split by 80:20 the data into training and validation sets.
Did stratified 5-fold cross validation on the training set.

#### Modeling

Used the xgboost's XGBClassifier (sklearn wrapper) with the following parameters:
**max_depth=3, n_estimators=700, learning_rate=0.05**