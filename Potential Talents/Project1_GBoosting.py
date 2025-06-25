#!/usr/bin/env python
# coding: utf-8

# ## Happy Customers  
# 
# #### May 2024

# #### Background:
# 
# We are one of the fastest growing startups in the logistics and delivery domain. We work with several partners and make on-demand delivery to our customers. From operational standpoint we have been facing several different challenges and everyday we are trying to address these challenges.
# 
# We thrive on making our customers happy. As a growing startup, with a global expansion strategy we know that we need to make our customers happy and the only way to do that is to measure how happy each customer is. If we can predict what makes our customers happy or unhappy, we can then take necessary actions.
# 
# Getting feedback from customers is not easy either, but we do our best to get constant feedback from our customers. This is a crucial function to improve our operations across all levels.
# 
# We recently did a survey to a select customer cohort. You are presented with a subset of this data. We will be using the remaining data as a private test set.
# 
# Data Description:
# 
# Y = target attribute (Y) with values indicating 0 (unhappy) and 1 (happy) customers
# X1 = my order was delivered on time
# X2 = contents of my order was as I expected
# X3 = I ordered everything I wanted to order
# X4 = I paid a good price for my order
# X5 = I am satisfied with my courier
# X6 = the app makes ordering easy for me
# 
# Attributes X1 to X6 indicate the responses for each question and have values from 1 to 5 where the smaller number indicates less and the higher number indicates more towards the answer.
# 

# In[2]:


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[3]:


# ## steps
# classification problem = gradient boosting
# 
# - 1. read data
# - 2. eda - visualize data, check dimension
# - 3. split features/labels
# - 4. build a pipeline: gradient boosting
# - 5. quick evaluation
# - 6. Permutation feature importance on the unseen test set
# - 7. Minimal feature subset that keeps ≥73 % accuracy

# In[5]:


# EDA
# 1. read the string as a DataFrame
df = pd.read_csv("ACME-HappinessSurvey2020.csv")
print((list(df.columns)))

# 2. split into features (X1–X6) and target (Y)
x_train = df[['X1', 'X2', 'X3', 'X4', 'X5','X6']].copy()
y_train = df['Y'].copy()

# quick sanity‑check
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)


# In[6]:


print("First five elements in X_train are:\n", x_train[:5])
print("Type of X_train:",type(x_train))


# In[40]:


print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))


# In[7]:


print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))


# ### Visualize data, 
# - check whether the target is skewed
# - histograms of each feature
# - boxplot of each feature by target class
# - correlation heat map among predictors
# 

# In[9]:


y_train.value_counts().plot(kind="bar")
plt.title("Class distribution in Y"); plt.xlabel("Class"); plt.ylabel("Count")


# In[10]:


x_train.hist(figsize=(10,6), bins=5)
plt.suptitle("Histograms of X1–X6", y=1.02)


# In[11]:


fig, axes = plt.subplots(1, 6, figsize=(15,4), sharey=True)
for ax, col in zip(axes, x_train.columns):
    sns.boxplot(x=y_train, y=x_train[col], ax=ax)
    ax.set_title(col)
    ax.set_xlabel("Y")
plt.suptitle("Distributions of X1–X6 split by class Y", y=1.05)


# In[12]:


# --- 3. Split features/labels ----------------------------------------------
X = df.drop(columns=["Y"])
y = df["Y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=768
)
print("Train shape :", X_train.shape, y_train.shape)
print("Test  shape :", X_test.shape,  y_test.shape)


# In[13]:


# --- 4. Build a pipeline: gradient boosting
model = GradientBoostingClassifier(random_state=768)
model.fit(X_train, y_train)



# In[14]:


# --- 5. Quick evaluation ----------------------------------------------------

pred = model.predict(X_test)

print(f"\nHold‑out accuracy: {accuracy_score(y_test, pred):.3%}")
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification report:\n", classification_report(y_test, pred, digits=3))


# In[15]:


# 6)  Permutation feature importance on the unseen test set
# ------------------------------------------------------------------
imp = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0)

importance = (
    pd.Series(imp.importances_mean, index=X.columns)
      .sort_values(ascending=False)
      .rename("Δ accuracy when shuffled")
)
print(importance.to_frame())


# In[16]:


# 7)  Minimal feature subset that keeps ≥73 % accuracy
#     (top‑k forward search in one line for brevity)
# ------------------------------------------------------------------
from itertools import combinations

best_subset, best_acc = None, 0
for k in range(1, 7):                         # try 1‑feature up to 6‑feature models
    for subset in combinations(importance.index, k):
        gb = GradientBoostingClassifier(random_state=768)
        gb.fit(X_train[list(subset)], y_train)
        acc = gb.score(X_test[list(subset)], y_test)
        if acc > best_acc:
            best_subset, best_acc = subset, acc
    if best_acc >= 0.73:                      # stop as soon as we clear the bar
        break

print(f"\n⚡  Best subset ≥73 %: {best_subset}  ->  {best_acc:.3%}")


# ### Final Notes
# 
# Gradient Boosting method seems to fit the model well compared to the logistic regression (various orders that I tried earlier)
# 
# The important features for predicting customers happiness appears to be X1, X5. This is consistent with the 'correct' weights piced by gradient  descent.  NB: Permuting a predictive future shoots up the mean absolute error.  
# 

# In[ ]:




