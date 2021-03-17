#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv('ab_data.csv')
df.head()


# In[3]:


df.info()


# b. Use the cell below to find the number of rows in the dataset.

# In[4]:


len(df)


# c. The number of unique users in the dataset.

# In[5]:


df.user_id.nunique()


# d. The proportion of users converted.

# In[6]:


df.converted.sum() / df.user_id.nunique()


# e. The number of times the `new_page` and `treatment` don't match.

# In[7]:


df.query('group == "treatment" or landing_page == "new_page"').shape[0] - df.query('group == "treatment" and landing_page == "new_page"').shape[0]


# f. Do any of the rows have missing values?

# In[8]:


df.isnull().sum()


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[9]:


treat_newp_xor = np.logical_xor((df.group == "treatment") , (df.landing_page == "new_page"))
df2 = df.drop(df[treat_newp_xor == True].index)


# In[10]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[11]:


df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[12]:


df2[df2.user_id.duplicated() == True]


# c. What is the row information for the repeat **user_id**? 

# In[13]:


df2[df2.user_id == 773192]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[14]:


df2.drop([1899], inplace = True)


# In[15]:


df2[df2.user_id.duplicated() == True]


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[16]:


df2.converted.sum()/len(df2)


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[17]:


# Setting masks
control = df2.group == 'control'
treatment = df2.group == 'treatment'


# In[18]:


df2[control].converted.mean()


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[19]:


df2[treatment].converted.mean()


# d. What is the probability that an individual received the new page?

# In[20]:


df2[df2.landing_page == 'new_page'].shape[0] / len(df2)


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# Examining the time interval for the test.

# In[21]:


df2.timestamp = pd.to_datetime(df2.timestamp)
df2.head()


# In[22]:


df2.info()


# **The test has been conducted on a fair number of individuals (290584 persons).<br> The ratio between the `treatment` and `control` groups was 1:1 , which is good.<br> Nevertheless, the probability of `converted` individuals among `treatment` group using `new_page` (11.88%) is marginally less than those in `control` group (12.04%) using the `old_page`.<br> There is not sufficient evidence that the `new_page` leads to more conversions.<br> It is noted that the test interval was about 22 days.**

# In[23]:


df2[treatment].groupby(df2[treatment].timestamp.dt.week).mean()


# In[24]:


df2[control].groupby(df2[control].timestamp.dt.week).mean()


# In[ ]:





# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **<div align="center">H0:  $p_{new}$  -  $p_{old}$  <=  0    <br>
#    H1: $p_{new}$  -  $p_{old}$  >  0     </div>**  
# Where H0 is the `Null Hypothesis` and H1 is the  `Alternative Hypothesis`

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[29]:


p_new = df2.converted.mean()
p_new


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# The null hypothesis stated above $p_{new}$ = $p_{old}$ = $p_{df2}$

# In[30]:


p_old = p_new
p_old


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[31]:


n_new = df2[treatment].shape[0]
n_new


# d. What is $n_{old}$, the number of individuals in the control group?

# In[32]:


n_old = df2[control].shape[0]
n_old


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[33]:


new_page_converted = np.random.choice([1,0], p=[p_new, 1-p_new], size = n_new)
new_page_converted


# **Note: this is similar to flipping a `loaded coin` with success rate equal to $p_{new}$ for $n_{new}$ times.**

# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[34]:


old_page_converted = np.random.choice([1,0], p=[p_old, 1-p_old], size = n_old)
old_page_converted


# **Note: this is similar to flipping a `loaded coin` with success rate equal to $p_{old}$ for $n_{old}$ times.**

# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[35]:


(new_page_converted.sum()/n_new) - (old_page_converted.sum()/n_old)


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[36]:


obs_diff = df2[treatment].converted.mean() - df2[control].converted.mean()
obs_diff


# In[37]:


p_diffs =np.empty(10000)
for x in range(10000):
    old_page_converted = np.random.choice([1,0], p=[p_old, 1-p_old], size = n_old)
    new_page_converted = np.random.choice([1,0], p=[p_new, 1-p_new], size = n_new)
    p_diffs[x] = (new_page_converted.sum()/n_new) - (old_page_converted.sum()/n_old)


# In[38]:


p_diffs.mean()


#  **As we sample the mean of differences so many times, their distribution tends toward a normal distribution as stated by `Central Limit Theorem`** 

# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[39]:


plt.hist(p_diffs);


# The method used in the hypothesis test is to model the null hypothesis and compare our observation to it.  
# In **(e, f)**, we tried to simulate our null hypothesis as if we were tossing two `biased coins` that have a success rate of the `conversion rate` in each group.  
# In **h** we did the `sampling distribution` (flipping our two loaded coins 10,000 times) where we get a normal distribution that is expressive to our null hypothesis.   
# The histogram above represents a normal distribution with a mean of 0 which is expected from the null hypothesis (in its extreme favor to the alternative hypothesis). The null hypothesis assumes there is no difference in probability to convertion rate from the `new_page` and the `old_page`

# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[34]:


plt.hist(p_diffs)
plt.axvline(obs_diff, c ='r')
plt.axvline(np.percentile(p_diffs, 97.5), c='g');


# The red line represents our observation and the green line represents the one-sided MOE (margin of error) of 5%.

# In[35]:


p_val = (p_diffs > obs_diff).mean()
p_val


# In[36]:


z_score = (obs_diff - p_diffs.mean())/ p_diffs.std()
z_score


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **`p_val` calculated above expresses the `P-Value` which is the the probability of observing our statistic `obs_diff` (or one more extreme in favor of the alternative) if the null hypothesis is true. P-Value here is more than the type-I error tolerance (0.05) which means we can not reject the null hypothesis. Thus no statistical evidence that there is a difference between the old and new pages regarding the conversion rate.**

# **Trying bootstrap technique**

# In[40]:


array_new = np.array(df2.query('group == "treatment"').converted)
array_old = np.array(df2.query('group == "control"').converted)
b_new = np.random.choice(array_new, 10000, replace = True)
b_old = np.random.choice(array_old, 10000, replace = True)
b_new


# In[38]:


b_diff = np.array(b_diff)
null_val = np.random.normal(0,b_diff.std(), 10000)
plt.hist(null_val)
plt.axvline(obs_diff, c = 'r')
plt.axvline(np.percentile(null_val, 97.5), c='g');


# In[39]:


b_diff.mean()


# In[40]:


pval_b = (null_val > obs_diff).mean()
pval_b


# In[41]:


z_score_b = (obs_diff - null_val.mean())/ null_val.std()
z_score_b


# **Boot strapping confirms our values in J section**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[44]:


import statsmodels.api as sm

convert_old = df2[control].converted.sum()
convert_new = df2[treatment].converted.sum()
count = np.array([convert_new, convert_old])
n_old = df2[control].shape[0]
n_new = df2[treatment].shape[0]
nobs = np.array([n_new, n_old])


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[45]:


stat, p_value = sm.stats.proportions_ztest(count, nobs, alternative='larger')


# In[46]:


stat, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# The `P-value` here is larger than the margin of error for `Type I Error` which is 0.05. This means we **fail to reject the null hypothesis.**  
# The `P-value` and `z-score` agree with **j, k** sections.  
# Z-score measures how far the observed value far from the mean in units of standard deviations. Z-score here means that our observation is not so far away from the mean and it is likely to come from this normal distribution under the null hypothesis conditions. This again means that we fail to reject the null and that the alternative hypothesis is not statistically significant

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Dealing with `Dichotomous variables` which alternate between only two values are best dealt with `Logistic Regression`**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[248]:


df2['intercept'] = 1
dummy= pd.get_dummies(df2.group)
dummy.head()


# In[249]:


df2['ab_page'] = dummy['treatment'] # As 1 when an individual receives the treatment and 0 if control.


# The `control` group shall be our `baseline`

# In[250]:


df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[251]:


log_m = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
res = log_m.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[252]:


print(res.summary2())


# Interpretation  of the Log Model:

# In[253]:


1/np.exp(-0.0150)


# This means that the probability of a converted individual to come from the `treatment` group (ab_page = 1) is 1/ exp(-0.015) or **1.015 times LESS** than the probability that the conversion rate from the `control` group.  
# This confirms the results from **Part II**

# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br>

# p-value = 0.1899  
# **The null hypothesis in the logistic regression model above is that there's no relationship between predicted and explanatory variables, while the alternative hypothesis is that there is a relationship.  
# The lower p-value here means we can assume a logistic relationship with less error. While in Part II, the lower p-value means we can assume the alternative hypothesis with less error.**

# ### Additional Test
# #### Model Diagnostics Using SCIKIT-Learn

# In[52]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split


# In[63]:


df2.head()


# In[171]:


df2_samp =[]


# In[185]:


df2_sam = df2.query('converted ==1').sample(20000)
df2_sam2 = df2.query('converted == 0').sample(10000)
df2_samp = df2_sam.append(df2_sam2)


# In[186]:


X = df2_samp[['ab_page']]
y= df2_samp['converted']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# In[187]:


log_mod = LogisticRegression()
log_mod.fit(X_train, y_train)
preds = log_mod.predict(X_test)
confusion_matrix(y_test, preds)


# In[188]:


tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
(tn, fp, fn, tp)


# In[189]:


log_mod.predict_proba(X_test)


# **The model fails to predict any true positive as the information based on new/old pages are `unbiased`. It means the lack of strong relation in the training set can't make our model learn how to predict which user is likely to convert or not in the test set.**

# ### Unbalanced Classes

# In[153]:


plt.hist(df2.converted);


# In[168]:


df2_samp =[]
df2_sam = df2.query('converted ==1').sample(100000, replace = True)
df2_sam2 = df2.query('converted == 0').sample(100000)
df2_samp = df2_sam.append(df2_sam2)


# In[155]:


df2_samp.head()


# In[156]:


plt.hist(df2_samp.converted);


# In[157]:


X = df2_samp[['ab_page']]
y= df2_samp['converted']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
log_mod = LogisticRegression()
log_mod.fit(X_train, y_train)
preds = log_mod.predict(X_test)
confusion_matrix(y_test, preds)


# In[158]:


precision_score(y_test, preds), recall_score(y_test, preds)


# In[159]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE()


# In[169]:


X = df2_samp[['ab_page']]
y= df2_samp['converted']
X, y = oversample.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
log_mod = LogisticRegression()
log_mod.fit(X_train, y_train)
preds = log_mod.predict(X_test)
confusion_matrix(y_test, preds)
precision_score(y_test, preds), recall_score(y_test, preds)


# In[ ]:





# In[ ]:





# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Other factors that may influence the test:**
# 1. Individuals' countries.
# 2. Individuals' ages.  
# 
# **Having more factors may enhance the quality of our model but it largely complicates the interpretation process and makes the results less intuitive.**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns. Provide the statistical output as well as a written response to answer this question.

# In[71]:


countries = pd.read_csv ('countries.csv')
countries.head()


# Wrangling over the dataframe

# In[72]:


countries.info()


# No missing values

# In[73]:


countries.country.unique()


# In[74]:


countries.user_id.nunique()


# Three countries exsist, thus we will have tree columns in the dummy (we will use two of them and the third will be our baseline.<br> but first will make sure that our index is the same in both dataframes, so each country goes right into its right user.

# In[75]:


countries = countries.set_index('user_id')
countries.head()


# Creating df3 database to merge it with the country

# In[76]:


df3 = df2.set_index('user_id')
df3.head()


# Joining the two dataframe

# In[77]:


df3 = df3.join(countries)


# In[78]:


df3.head()


# Creating dummies for the country column

# In[79]:


dummy = pd.get_dummies(df3.country, drop_first= True)
dummy.head()


# In[80]:


df3 = df3.join(dummy)
df3.head()


# Using `CA` Canada as baseline for the `countries`

# In[81]:


logit_mod = sm.Logit(df3['converted'], df3[['intercept','ab_page', 'UK', 'US']])
res2 = logit_mod.fit()
print(res2.summary2())


# In[82]:


np.exp(-0.0149 ), np.exp(0.0506), np.exp(0.0408)


# In[83]:


df3_samp =[]
df3_sam = df3.query('converted ==1').sample(100000, replace = True)
df3_sam2 = df3.query('converted == 0').sample(100000)
df3_samp = df3_sam.append(df3_sam2)


# In[84]:


X = df3_samp[['ab_page', 'UK', 'US']]
y= df3_samp['converted']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
log_mod = LogisticRegression()
log_mod.fit(X_train, y_train)
preds = log_mod.predict(X_test)
confusion_matrix(y_test, preds)
precision_score(y_test, preds), recall_score(y_test, preds)


# In[85]:


confusion_matrix(y_test, preds)


# In[ ]:





# In[ ]:





# **We can say the odds of individuals from UK is 1.05 times more to convert relating to those from Canada.
# Also, individuals from US is 1.04 times more likely to convert relating to those from Canada.**

# **All coefficient values in the above model (when exponentiated) is near to one, which means it doesn't have a *practical* significance ratio among `converted` individuals group.**

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[86]:


df3['UK_treatment'] = df3['UK'] * df3['ab_page'] # Individuals from UK and in the treatment group
df3['US_treatment'] = df3['US'] * df3['ab_page'] # Individuals from US and in the treatment group
df3.head()


# In[87]:


log_mod2 = sm.Logit(df3['converted'], df3[['intercept', 'UK_treatment', 'US_treatment', 'UK', 'US']])
res2 = log_mod2.fit()
print(res2.summary2())


# In[88]:


np.exp(0.0901), np.exp(0.0644)


# **From above we can infer that the conversion from the `new page` is slightly favorable in UK.**

# In[89]:


df3.UK.sum()/len(df2) # precentage of users from UK


# But we can see that users from UK is barely 25% of our database. This is why it did not had a great impact on the overall page conversions.

# **Although the `P-values` for the model have improved but the coefficients haven't greatly increased. Interpreting the model is not intuitive but no change in the coefficient value exponentiated.**

# In[215]:


df3_samp =[]
df3_sam = df3.query('converted ==1').sample(100000, replace = True)
df3_sam2 = df3.query('converted == 0').sample(100000)
df3_samp = df3_sam.append(df3_sam2)


# In[216]:


X = df3_samp[['ab_page', 'UK', 'US', 'UK_treatment', 'US_treatment' ]]
y= df3_samp['converted']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
log_mod = LogisticRegression()
log_mod.fit(X_train, y_train)
preds = log_mod.predict(X_test)
confusion_matrix(y_test, preds)
precision_score(y_test, preds), recall_score(y_test, preds)


# In[217]:


confusion_matrix(y_test, preds)


# In[220]:


log_mod2 = sm.Logit(df3_samp['converted'], df3_samp[['intercept','UK', 'US']])
res2 = log_mod2.fit()
print(res2.summary2())


# ### Examining the time variables

# Here I would like to look if the likelihood of conversion from `treatment` group is increasing with time. If so we can recommend the continue the test till we get a converged result.

# In[197]:


df4 = df3


# In[198]:


df4['week'] = df4.timestamp.dt.week # extracting the week number from the timestamp column


# In[199]:


dummy = pd.get_dummies(df4.week, drop_first = True) #creating dummy coloums


# In[200]:


dummy.head()


# In[201]:


df4[['2', '3', '4']] = dummy #pasting the dummies in the dataframe


# In[211]:


df4_samp =[]
df4_sam = df4.query('converted ==1').sample(100000, replace = True)
df4_sam2 = df4.query('converted == 0').sample(100000)
df4_samp = df4_sam.append(df4_sam2)


# In[212]:


#logistic regression classification
logit_mod = sm.Logit(df4_samp['converted'], df4_samp[['intercept','ab_page' ,'2', '3', '4']])
res2 = logit_mod.fit()
print(res2.summary2())


# In[213]:


np.exp(0.0018), np.exp(0.0195), np.exp(0.0328)


# The p-values and coefficients indicate week relationships. This means no strong relation with time and there is no need to keep going with the test.

# In[130]:


df4_samp =[]
df4_sam = df4.query('converted ==1').sample(30000)
df4_sam2 = df4.query('converted == 0').sample(30000)
df4_samp = df4_sam.append(df4_sam2)


# In[172]:


X = df4_samp[['ab_page', 'UK', 'US', 'US_treatment','UK_treatment', 'week' ]]
y= df4_samp['converted']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
log_mod = LogisticRegression()
log_mod.fit(X_train, y_train)
preds = log_mod.predict(X_test)
confusion_matrix(y_test, preds)
precision_score(y_test, preds), recall_score(y_test, preds)


# In[173]:


X_test


# In[ ]:





# In[ ]:





# <a id='conclusions'></a>
# ## Conclusions:
# In this model we can conclude that the `new_page` has failed to show a significance effect upon the conversion rate. Using more than one modelling method, we found that the odds of conversion from `new_page` is **not better** than the `old_page`.<br> By examining the time variables I saw no optimistic clue to run the test for even further more. 
# 
# 
# **Notes:**
# * We have removed 3893 records that showed incosistance data.
# 
# 

# In[80]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




