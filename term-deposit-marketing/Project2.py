
# # Project 2
# 
# ## Term Deposit Marketing
# 
# #### May 2024

# #### Background:
# 
# We are a small startup focusing mainly on providing machine learning solutions in the European banking market. We work on a variety of problems including fraud detection, sentiment classification and customer intention prediction and classification.
# 
# We are interested in developing a robust machine learning system that leverages information coming from call center data.
# 
# Ultimately, we are looking for ways to improve the success rate for calls made to customers for any product that our clients offer. Towards this goal we are working on designing an ever evolving machine learning product that offers high success outcomes while offering interpretability for our clients to make informed decisions.
# 
# Data Description:
# 
# The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.
# 
# Attributes:
# 
#    - age : age of customer (numeric)
# 
#    - job : type of job (categorical)
# 
#    - marital : marital status (categorical)
# 
#    - education (categorical)
# 
#    - default: has credit in default? (binary)
# 
#    - balance: average yearly balance, in euros (numeric)
# 
#    - housing: has a housing loan? (binary)
# 
#    - loan: has personal loan? (binary)
# 
#    - contact: contact communication type (categorical)
# 
#    - day: last contact day of the month (numeric)
# 
#    - month: last contact month of year (categorical)
# 
#    - duration: last contact duration, in seconds (numeric)
# 
#    - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# Output (desired target):
# 
# y - has the client subscribed to a term deposit? (binary)
# 
# Goal(s):
# 
# Predict if the customer will subscribe (yes/no) to a term deposit (variable y)
# 
# Success Metric(s):
# 
# Hit %81 or above accuracy by evaluating with 5-fold cross validation and reporting the average performance score.
# 
# Current Challenges:
# 
# We are also interested in finding customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize.
# 
# What makes the customers buy? Tell us which feature we should be focusing more on.
# 

# In[154]:


import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance

get_ipython().run_line_magic('matplotlib', 'inline')


# ## steps
# classification problem = logistic regression and random forest
# 
# - 1. read data & eda - visualize data, check dimension
# - 2. EDA - visualize data, check dimensions
# - 3. Split features/labels (stratify the split to keep the yes/no ratio the same in train and test)
# - 4. Build the pipelines: logistic regression and random forest
# - 5. Evaluate baseline performance via stratified 5-fold CV
# - 6. Model valuation: cross-validation set and test set
# - 7. Determine the segment(s) of customers our client should prioritize.
#     aize.
#         - fit on full data and get predicted probabilities + default predictions
#         - segment “likely yes” customers (prediction==1)    
#         - rank segments by average predicted probabilities
# - 8. A feature to focus on.
# - 9. Use feature importances & permutation importance
# - Others
# - 10. Compare results from this Logistic Regression model with those from Random Forest
# - 11. Final comments
# 

# In[86]:


# EDA
# 1. read the file as a DataFrame
df = pd.read_csv("term-deposit-marketing-2020.csv")
print((list(df.columns)))


# In[87]:


# split into features (X) and target (Y)
x_ = df[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign']].copy()
y_ = df['y'].copy()

# quick sanity‑check
print("x_ shape:", x_.shape)
print("y_ shape:", y_.shape)


# In[88]:


print("First five elements in X_ are:\n", x_[:5])
print("Type of X_:",type(x_))


# In[89]:


print("First five elements in y_ are:\n", y_[:5])
print("Type of y_:",type(y_))


# In[90]:


print("\nMissing values per column:")
print(df.isnull().sum())


# In[91]:


print("\nNumeric summary:")
print(df.describe())


# In[92]:


# Boxplots to spot outliers
numeric_cols = ['age','balance','duration','campaign']
for col in numeric_cols:
    plt.figure()
    plt.boxplot(df[col].dropna(), vert=False)
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.tight_layout()


# In[93]:


# Categorical counts
for col in ['job','marital','education','default','housing','loan','contact','y']:
    counts = df[col].value_counts()
    print(f"\n{col} value counts:")
    print(counts)


# ### Visualize data, 
# - check whether the target is skewed
# - histograms of each feature
# - boxplot of each feature by target class
# 

# In[95]:


y_.value_counts().plot(kind="bar")
plt.title("Class distribution in Y"); plt.xlabel("Class"); plt.ylabel("Count")


# In[96]:


x_.hist(figsize=(10,10), bins=5)
plt.suptitle("Histograms of the numeric features", y=1.02)


# In[97]:


# Bar‐plots for categorical variables
for col in ['job','marital','education','default','housing','loan','contact']:
    plt.figure()
    df[col].value_counts().plot(kind='bar')
    plt.title(f'{col} distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()


# In[98]:


# Identify columns; map yes, no in y to 1,0
y_ = df["y"].map({"no": 0, "yes": 1})


numeric_cols = ["age", "balance", "day", "duration", "campaign"]
categorical_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])
#with scaling, the numeric features enter the model with mean 0 and variance 1, 
#ensuring fair regularization and more reliable convergence—typically boosting both accuracy and minority-class recall.


# In[99]:


#  2. Split features/labels: train, cross validation and test sets

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
x_bc_train, x__, y_bc_train, y__ = train_test_split(x_, y_, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_bc_cv, x_bc_test, y_bc_cv, y_bc_test = train_test_split(x__, y__, test_size=0.50, random_state=1)

# Delete temporary variables
del x__, y__

print(f"the shape of the training set (input) is: {x_bc_train.shape}")
print(f"the shape of the training set (target) is: {y_bc_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_bc_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_bc_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_bc_test.shape}")
print(f"the shape of the test set (target) is: {y_bc_test.shape}")


# In[100]:


#3. Build a pipeline: LogisticRegression
model = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(
        solver='liblinear',
        class_weight='balanced', 
        max_iter=10000))
    ])

#class_weight='balanced': automatically reweights the minority class higher to combat imbalance.
#5-fold stratified cross-validation

#4. evaluate baseline performance via stratified 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, x_bc_train, y_bc_train, cv=cv, scoring='accuracy')
print(f"5‐fold CV accuracy (balanced LR): {cv_scores.mean():.4f}")


# In[101]:


model.fit(x_bc_train, y_bc_train)


# In[102]:


# 5. Model valuation: cross-validation set

pred = model.predict(x_bc_cv)

print(f"\nHold‑out accuracy: {accuracy_score(y_bc_cv, pred):.3%}")
print("\nConfusion matrix:\n", confusion_matrix(y_bc_cv, pred))
print("\nClassification report:\n", classification_report(y_bc_cv, pred, digits=3))


# In[103]:


# Model valuation: test set

pred_test = model.predict(x_bc_test)

print(f"\nHold‑out accuracy: {accuracy_score(y_bc_test, pred_test):.3%}")
print("\nConfusion matrix:\n", confusion_matrix(y_bc_test, pred_test))
print("\nClassification report:\n", classification_report(y_bc_test, pred_test, digits=3))


# In[127]:


# 5. a. Fit on full data and get predicted probabilities + default predictions
model.fit(x_, y_)
probs = model.predict_proba(x_)[:, 1]
preds = (probs >= 0.5).astype(int)  # default threshold = 0.5

# b. Segment “likely yes” customers (pred == 1)
seg = df[preds == 1].copy()
seg['pred_prob'] = probs[preds == 1]

# c. Rank segments by average predicted probability
for col in ['job', 'education', 'marital']:
    grouping = (
        seg.groupby(col)['pred_prob']
           .agg(['count', 'mean'])
           .sort_values('mean', ascending=False)
           .head(5)
    )
    print(f"\nTop 5 segments by {col}:")
    display(grouping)


# In[130]:


# 7. Permutation Importance
perm_imp = permutation_importance(
    estimator=model,
    X=x_bc_test,
    y=y_bc_test,
    n_repeats=10,
    random_state=42,
    scoring='accuracy'
)

# ORGANIZE RESULTS INTO A DATAFRAME 
# `importances_mean` and `importances_std` are arrays of length = number of original features.
feature_names = x_bc_test.columns.tolist()
imp_df = pd.DataFrame({
    'feature'         : feature_names,
    'mean_importance' : perm_imp.importances_mean,
    'std_importance'  : perm_imp.importances_std
})

# Sort descending by mean_importance
imp_df = imp_df.sort_values(by='mean_importance', ascending=False)

# DISPLAY TOP 10 FEATURES BY PERMUTATION IMPORTANCE 
top10 = imp_df.head(10).reset_index(drop=True)
print("\nTop 10 Features by Permutation Importance (drop in Accuracy):")
print(top10.to_string(index=False))

# Plot
plt.figure()
plt.bar(imp_df['feature'], imp_df['mean_importance'])
plt.xticks(rotation=90)
plt.title("Permutation Importance")
plt.tight_layout()
plt.show()


# In[132]:


perm_imp = permutation_importance(
    estimator=model,
    X=x_bc_test,
    y=y_bc_test,
    n_repeats=10,
    random_state=42,
    scoring='recall'
)

# ORGANIZE RESULTS INTO A DATAFRAME
# `importances_mean` and `importances_std` are arrays of length = number of original features.
feature_names = x_bc_test.columns.tolist()
imp_df = pd.DataFrame({
    'feature'         : feature_names,
    'mean_importance' : perm_imp.importances_mean,
    'std_importance'  : perm_imp.importances_std
})

# Sort descending by mean_importance
imp_df = imp_df.sort_values(by='mean_importance', ascending=False)

# DISPLAY TOP 10 FEATURES BY PERMUTATION IMPORTANCE
top10 = imp_df.head(10).reset_index(drop=True)
print("\nTop 10 Features by Permutation Importance (drop in Accuracy):")
print(top10.to_string(index=False))

# Plot
plt.figure()
plt.bar(imp_df['feature'], imp_df['mean_importance'])
plt.xticks(rotation=90)
plt.title("Permutation Importance")
plt.tight_layout()
plt.show()


# In[134]:


perm_imp = permutation_importance(
    estimator=model,
    X=x_bc_test,
    y=y_bc_test,
    n_repeats=10,
    random_state=42,
    scoring='f1'
)

# ORGANIZE RESULTS INTO A DATAFRAME
# `importances_mean` and `importances_std` are arrays of length = number of original features.
feature_names = x_bc_test.columns.tolist()
imp_df = pd.DataFrame({
    'feature'         : feature_names,
    'mean_importance' : perm_imp.importances_mean,
    'std_importance'  : perm_imp.importances_std
})

# Sort descending by mean_importance
imp_df = imp_df.sort_values(by='mean_importance', ascending=False)

# DISPLAY TOP 10 FEATURES BY PERMUTATION IMPORTANCE
top10 = imp_df.head(10).reset_index(drop=True)
print("\nTop 10 Features by Permutation Importance (drop in Accuracy):")
print(top10.to_string(index=False))

# Plot
plt.figure()
plt.bar(imp_df['feature'], imp_df['mean_importance'])
plt.xticks(rotation=90)
plt.title("Permutation Importance")
plt.tight_layout()
plt.show()


# ### Observations:
# #### EDA
# - Y column skewed (the target y contains more 'no' and 'yes' values. Therefore, the data is imbalanced.
# - X columns are a combination of numerical and categorical columns.
# - No missing values.
# - 3 top jobs: blue collar, management, technician.
# 
# The Logistic Regression method seems to fit the model well. I also plan to fit the data using the Random Forest Model. It's essential to note that the data is limited, with a high ratio of 'No' versus 'Yes', which impacted the results, particularly recall.
# 
# To improve recall, the following was included:
# - a standard scaler for the numerical columns.
# - StratifiedKFold in the built-on-the- the fly cross validation.
# - class_weight='balanced' in the model. This automatically reweights the minority class higher to combat imbalance.
# 
# This improved the result. The 5-fold CV accuracy (balanced LR) achieves a score of 86.7%. This exceeds the 81% specified in the requirements. Additionally, the hold-out accuracy for the model is 87.3 on the CV dataset and 86.9 on the test dataset. Observation: The model achieves a high recall (85%), but its precision (33%) and F1 score (48%) are low for the minority class. We may need to apply the SMOTE technique to address the imbalance in the dataset.
# 
# Determine the segment(s) of customers our client should prioritize.
#  
# - Top 5 segments by job: housemaid, self-employed, unemployed, management, technician.
# - Top 5 segments by education: primary, unknown, tertiary, secondary.
# - Top 5 segments by marital status: divorced, married, single.
# The results from the permutation importance indicate that the fop feature to focus on is the “last contact duration.
# 
# ### Other observations
# Results from the Random Forest Model (not included in this notebook) indicate a higher precision (60%) score for the minority class, but a significantly lower recall (15%) and F-score (25%). The choice of whether to prioritize recall or precision depends on the cost per contact and the value of a converted subscriber. If the cost per customer is similar to or exceeds the value a converted subscriber brings, then precision should be prioritized.
# 
# The Top 5 segments by job are different for the RF model compared to what the LR model suggested. 
# - Top 5 segments by job: retired, management, services, student, technician.
# - Top 5 segments by education: tertiary, secondary, primary,  unknown.
# - Top 5 segments by marital status:  single, divorced, married.
# 
# The results from the permutation importance indicate that the feature to focus on is the “last contact duration."
# 
# 

# In[ ]:




