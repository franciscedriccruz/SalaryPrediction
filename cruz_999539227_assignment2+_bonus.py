
# coding: utf-8

# # MIE1624 Assignment 2 - Salary Prediction - Bonus

# # Francis Cruz - 999539227

# ### Section Headers: 
# 
# #### 0.0. Introduction and Importing Libraries
# - 0.1. Importing Libraries
# 
# - 1.0. Data Cleaning
# - 1.1. Import Dataset
# - 1.2. Data Cleaning and Preparation
# - 1.2.1. Cleaning Numerical Entries in Dataframe
# - 1.2.2. Missing Data Cleaning
# - 1.3. Feature Engineering (Countries to Continents)
# - 1.4. Encoding Categorical Data
# 
# #### 2.0 Exploratory Analysis
# - 2.1 Graphical Figures Depicting Trends in the Data
# - 2.1.a. Trends on Yearly Salaries vs Continent of Residence and Age
# - 2.1.b. Trends on Yearly Salaries vs Education and Years of Experience
# - 2.1.c. Trends on Yearly Salaries vs Programming Proficiency
# - 2.1. Summary
# - 2.2. Feature Importance
# 
# #### 3.0. Feature Engineering and Selection
# - 3.1. Feature Engineering
# - 3.1.a. Continent of Residence
# - 3.1.b. Combine Part Questions
# - 3.1.c. Target Variable Manipulation (Remove Outliers and SQRT Transform)
# - 3.1 d. Scaling Features
# - 3.2. Feature Selection
# - 3.2.1. Feature Selection: Lasso Regularization
# - 3.3. Modifying Dataset with Selected Features
# - 3.3.1. Feature Importance of Selected Features
# - 3.0 Summary
# 
# #### 4.0 Models Selected from Assignment
# - 4.1. Lasso Linear Regression
# - 4.2. Gradient Boosting Regression
# - 4.3. Random Forest Regression
# - 4.4. Epsilon-Support Vector Regression
# - 4.5. Combine Metrics to Compare to Neural Network
# 
# #### 5.0. Neural Network Implementation - BONUS
# - 5.1. Parametric Study - Number of Hidden Layers and Nodes per layer
# - 5.2. Parametric Study - Momentum
# - 5.3. Parametric Study - Learning Rate
# - 5.4. Parametric Study - Number of Iterations
# - 5.5. Optimal Neural Network
# 
# #### 6.0. Comparison to Other Models - BONUS

# # 0.0 Introduction and Importing Libraries

# The objective of the bonus is to implement a neural network to predict the salary of an individual based on the responses from the Kaggle survey. Moreover, a discussuion on how the performance of the neural network changes with different parameters is also presented. See section 5.0 onwards. 

# # 0.1 Importing Libraries

# In[1]:


# Import Libraries
import seaborn as sns
import numpy as np            
import pandas as pd            
from pandas import read_csv
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPRegressor


# # 1.0 Data Cleaning

# In this section, we will first look into the raw data set to see any issues with the data. 

# ## 1.1 Import Dataset

# In[2]:


# Import dataset from Kaggle competition
raw_data = pd.read_csv("Kaggle_Salary.csv", header=0) 


# In[3]:


# Take a look at the dataset
raw_data.head(n=5)


# ## 1.2 Data Cleaning and Preparation 

# Based on the above dataset, the first row contains the question in the Kaggle survey. We will store these questions in case we need this information again. Additionally, there are entries that contain text answers, typically for "Others" questions. For ease of data analysis, we will drop these columns and focus on the fixed selections in the survey (i.e. drop down questions, multiple choice, etc.). Text responses in this survey will likely be very different from one another even if CountVectorization was used. Moreover, implementing text features in the model will increase the computational complexity of the model, thus making it difficult to train and test multiple times, particularly when performing k-fold cross validation. 
# 
# ### Provide Insight on why you think missing values are missing and how your approach might impact the overall analysis: 
# Missing information in the form of NaNs were seen particularly for questions that required a fixed response (i.e. multiple choice or drop down questions). These missing bvalues appeared when the user did not select this choice as their response in the survey (specific choice did not apply to them). Apart from this instance, NaNs may also appear if the user did not enter any answer to optional questions or even in the provided text boxes in the survey. The answers of these surveys appear as new columns per option for each question. Despite missing values, these values need to be replaced with an integer or fixed categorical variable to be useful for data analysis. When fixing the data is complete, these columns may become useful for the modelling aspect of this problem depending on its feature importance level. Moreover, some missing values were also replaced with the mode of the column. While this is not ideal, it would allow us to have a larger sample set to use to model with the assumption that the most common answer would most likely be the answer for those missing values in the same column. 

# In[4]:


# Print what are the column names
list(raw_data.columns.values)


# In[5]:


# Drop entries containing free text
colsToDrop = ['Unnamed: 0',
 'Time from Start to Finish (seconds)',
 'Q1_OTHER_TEXT',
 'Q6_OTHER_TEXT',
 'Q7_OTHER_TEXT',
 'Q11_OTHER_TEXT',
 'Q12_OTHER_TEXT',
 'Q13_OTHER_TEXT',
 'Q14_OTHER_TEXT',
 'Q15_OTHER_TEXT',
 'Q16_OTHER_TEXT',
 'Q17_OTHER_TEXT',
 'Q18_OTHER_TEXT',
 'Q19_OTHER_TEXT',
 'Q20_OTHER_TEXT',
 'Q21_OTHER_TEXT',
 'Q22_OTHER_TEXT',
 'Q27_OTHER_TEXT',
 'Q28_OTHER_TEXT',
 'Q29_OTHER_TEXT',
 'Q30_OTHER_TEXT',
 'Q31_OTHER_TEXT',
 'Q32_OTHER',
 'Q33_OTHER_TEXT',
 'Q34_OTHER_TEXT',
 'Q35_OTHER_TEXT',
 'Q36_OTHER_TEXT',
 'Q37_OTHER_TEXT',
 'Q38_OTHER_TEXT',
 'Q42_OTHER_TEXT',
 'Q49_OTHER_TEXT',
 'Q50_OTHER_TEXT', 
 'index']

raw_data = raw_data.drop(colsToDrop, axis=1)


# In[6]:


# Drop first row but keep to properly label columns later
question_column_labels = raw_data.index[0]
raw_data=raw_data.drop(raw_data.index[0])
raw_data=raw_data.reset_index(drop=True)
raw_data.head(5)


# In[7]:


# Check data types in each columns
print(raw_data.dtypes)


# ## 1.2.1 Cleaning Numerical Entries in Dataframe

# By inspecting the dataframe, questions 9, 34, and 35 contain numerical entries but are currently object types. Question 9 contains the salary information, Q34 % time spent in a data science project, and Q35 % of time spent in data science training. We will convert these columns into numerical types. 

# In[8]:


print(len(raw_data))


# In[9]:


# Question 9, 34, and 35 contain numbers but are currently object types
# Question 9 - Salary
# Question 34 - % of time spent in a data science project 
# Question 35 - % of time spent in data science training

for column in raw_data.columns.values:
    if "Q9" in column or "Q34" in column or "Q35" in column:
        raw_data[column] = pd.to_numeric(raw_data[column])
# raw_data['Q9'] = pd.to_numeric(raw_data['Q9'])
raw_data.head()


# In[10]:


# Print data types
print(raw_data.dtypes)


# We need to ensure that all the numerical values in Q34 and Q35 makes sense. We will first look into Q34 to see if all the percentages provided sums to a total of 100%. A similar approach will be performed for Q35. 

# In[11]:


# store all current columns pertaining to both Q34 and Q35
Q34_parts = []
Q35_parts = []
for column in raw_data.columns.values:
    if "Q34" in column:
        Q34_parts.append(column)
    elif "Q35" in column:
        Q35_parts.append(column)
    else:
        continue
print(Q34_parts)
print(Q35_parts)


# In[12]:


# let's add all of the columns for each row pertaining to Q34 and Q35
raw_data['Sum_Q34'] = raw_data[Q34_parts].sum(axis=1)
raw_data['Sum_Q35'] = raw_data[Q35_parts].sum(axis=1)


# In[13]:


raw_data


# In[14]:


plt.hist(raw_data['Sum_Q34'], edgecolor='black')
plt.title('Count of Total Percentages in Q34 Before Cleaning')
plt.ylabel('count')
plt.xlabel('Q34 Percentage')
plt.show()


# In[15]:


plt.hist(raw_data['Sum_Q35'], edgecolor='black')
plt.title('Count of Total Percentages in Q35 Before Cleaning')
plt.ylabel('count')
plt.xlabel('Q35 Percentage')
plt.show()


# We can see that not all of them summed to 100% and some did not even fill in an answer for Q34 and Q35. A potential reason for not having them close to 100% is to due to the optional answers for each question (i.e. 'Q34_OTHER_TEXT', 'Q35_OTHER_TEXT'). Some individuals may enter a value in these columns. However, these columns were dropped as it contained text answers and may complicate the model. Hence, we will renormalize the existing percentages for Q34 and Q35 and drop all rows that contained a sum less than 60%. 

# In[16]:


# Drop colums if sum < 60 or is nan
raw_data = raw_data[raw_data['Sum_Q34'] > 60]
raw_data = raw_data[raw_data['Sum_Q35'] > 60]


# In[17]:


raw_data


# In[18]:


plt.hist(raw_data['Sum_Q34'], edgecolor='black')
plt.title('Count of Total Percentages in Q34 After Cleaning')
plt.ylabel('count')
plt.xlabel('Q34 Percentage')
plt.show()


# In[19]:


plt.hist(raw_data['Sum_Q35'], edgecolor='black')
plt.title('Count of Total Percentages in Q35 After Cleaning')
plt.ylabel('count')
plt.xlabel('Q35 Percentage')
plt.show()


# Now that all of the summed percentages are non-zero and are not NaN, we can renormalize the parts column for both Q34 and Q35 so that adding each of them will result in 100%. 

# In[20]:


raw_data[Q34_parts].head(5)


# In[21]:


raw_data[Q34_parts] = raw_data[Q34_parts].div(raw_data.Sum_Q34, axis=0)
raw_data[Q34_parts] = raw_data[Q34_parts].multiply(100)
raw_data[Q34_parts].head(5)


# In[22]:


raw_data[Q35_parts].head(5)


# In[23]:


raw_data[Q35_parts] = raw_data[Q35_parts].div(raw_data.Sum_Q35, axis=0)
raw_data[Q35_parts] = raw_data[Q35_parts].multiply(100)
raw_data[Q35_parts].head(5)


# We need to double check each of the entries of the parts for Q34 and Q35 to ensure that there were no negative numbers entered. 

# In[24]:


raw_data[Q34_parts].describe()


# In[25]:


raw_data[Q35_parts].describe()


# We can see that there was a negative entry in Q35_Part_5, we will remove this row as it does not contain any reasonable data. 

# In[26]:


raw_data = raw_data[raw_data['Q35_Part_5'] >= 0]


# In[27]:


raw_data[Q35_parts].describe()


# In a similar manner, we will perform a quick sanity check on the salary entries in the remaining data set. 

# In[28]:


# Print quick statistics on the salary information in the dataset
raw_data['Q9'].describe()


# A quick check on the data revealed that someone entered 16 USD as their annual salary! Let's sort through the values by salary to see if other people earned similar amounts and if it seems reasonable. 

# In[29]:


salary_check = raw_data.sort_values(by='Q9')
salary_check.head(10)


# A quick look showed that some individuals earn significantly less especially for those who reside in India, Brazil, China, etc. Some of these entries are unreasonable as an individual in the US only earned USD$25 for the entire year. Let's plot a histogram of salaries for those who earned less than 5000. 

# In[30]:


salary_check = salary_check[salary_check['Q9'] < 1000]


# In[31]:


plt.hist(salary_check['Q9'], edgecolor='black')
plt.title('Count of Individuals who earned less than 1000 a year')
plt.ylabel('count')
plt.xlabel('Salary (USD$)')
plt.show()


# While a smaller salary is expected for those living in developing countries, there are some outliers as indicated in the dataframe above (for those who live in the US and only earned $25). Hence, some entries such as salaries less than 500 dollars a year is unreasonable. We will remove these entries in the original dataset. 

# In[32]:


raw_data = raw_data[raw_data['Q9'] > 500]
raw_data.describe()


# In[33]:


# lets drop the Sum_Q34 and Sum_Q35 columns as they are no longer needed. 
raw_data = raw_data.drop(['Sum_Q34', 'Sum_Q35'], axis=1)


# In[34]:


len(raw_data)


# We are now ready to proceed with dealing with missing data, particularly in the categorical data columns. 

# ## 1.2.2 Missing Data Cleaning

# In this section, we address the issue of missing data. Following the assignment's recommendation, we will use the mode of each column to fill the missing data. While this is not ideal, it is one method to address missing data by filling these voids will data that is common with the dataset. Doing so presents some inaccuracies but is required to have a sufficently high number of samples to train our models. 

# In[35]:


# Determine which columns (i.e. questions) contain missing data
missing_data = raw_data.isnull().sum(axis=0)
missing_data[missing_data > 0]


# If a column has more than 95% missing data, then this column does not present any useful information. Hence, this column is dropped. 

# In[36]:


# type(missing_data)
colsToDrop = []
for index, value in missing_data.items():
    if value > 0.95*len(raw_data):
        colsToDrop.append(index)


# In[37]:


# if the column had 95% of its entries with missing data, then drop the column
raw_data = raw_data.drop(colsToDrop, axis=1)


# Multiple questions contained parts. Hence, we can create dummies to represent answers based on the number of unique available answers per question. This will aid in filling missing information, particularly for the fixed selection questions (as indicated before).  

# In[38]:


# Create dummies for "Part" Questions
for column in raw_data.columns.values:
    num_unique_options = len(raw_data[column].unique())
    
    if "Part" in column:
        if num_unique_options <= 1:
            raw_data[column] = 0
        else:
            raw_data[column] = pd.get_dummies(raw_data[column])


# In[39]:


# After creating dummy variables, print any other missing information columns
missing_data = raw_data.isnull().sum(axis=0)
missing_data[missing_data > 0]


# In[40]:


# A brief desciption of these questions are listed below: 
# Questions that containing missing information: 
# Question 5 asks about undergraduate major
# Question 8 asks about years of experience
# Question 17 programming language (cat)
# Question 18 recommended programming language to beginners (cat)
# Question 20 ML language used the most (cat)
# Q22 which data visualization do you use the most (cat)
# Q32 What type of data do you use (cat)
# Q37 Online platform (cat)
# Q40 Which better demonstrates experistise in data sci 
# Q43 Approx what tme of your projects involve exploring bias
# Q46 Approx what tme of your projects involve exploring insights
# Q48 Do you consider ML as blackboxes


# In[41]:


# Fill missing information with modes instead. 
def fillNaNwithMode(dataframe, column):
    dataframe[column].fillna(dataframe[column].mode()[0],inplace=True)
    return dataframe


# In[42]:


for element in np.arange(len(missing_data)):
    raw_data = fillNaNwithMode(raw_data, missing_data.index[element])


# In[43]:


# Check if we still have any missing information: 
missing_data = raw_data.isnull().sum(axis=0)
missing_data[missing_data > 0]


# In[44]:


raw_data.head(5)


# Noticeably, a lot of the entries in the data set contain categorical data. Thus, we need to use an encoder to convert text data into numbers, which predictive models can better understand. We will be encoding these categorical data after the following subsection. 

# ## 1.3 Feature Engineering (Countries to Continents)

# Prior to encoding, a particular column of interest is Q3 - indicating the country of residence of the user. There are multiple unique countries listed in this column. To aid with this analysis, these countries were converted to continents of residence to reduce the unique entries in these columns. Feature Engineering was performed in this step prior to section 3.0 as performing the following tasks here will aid with visualization in the Exploratory Analysis in section 2.0.  

# In[45]:


# Remove no information Q3 data
raw_data = raw_data[raw_data['Q3'] != 'Other']
raw_data = raw_data[raw_data['Q3'] != 'I do not wish to disclose my location']


# In[46]:


# Country Dictionary Mapping to Continet
country_dict = {
 'United States of America':'North America',
 'India':'Asia',
 'Chile':'South America',
 'Hungary':'Europe',
 'France':'Europe',
 'Argentina':'South America',
 'Japan':'Asia',
 'Colombia':'South America',
 'Nigeria':'Africa',
 'Spain':'Europe',
 'Iran, Islamic Republic of...': 'Middle East',
 'United Kingdom of Great Britain and Northern Ireland': 'Europe',
 'Turkey': 'Middle East',
 'Poland':'Europe',
 'Kenya':'Africa',
 'Denmark':'Europe',
 'Netherlands':'Europe',
 'Sweden':'Europe',
 'Ukraine':'Europe',
 'Canada': 'North America',
 'Australia': 'Oceania',
 'Russia':'Europe',
 'Italy':'Europe',
 'Mexico':'North America',
 'Germany':'Europe',
 'Singapore':'Asia',
 'Indonesia':'Asia',
 'Brazil': 'South America',
 'China':'Asia',
 'South Africa':'Africa',
 'South Korea':'Asia',
 'Malaysia':'Asia',
 'Hong Kong (S.A.R.)':'Asia',
 'Portugal':'Europe',
 'Thailand':'Asia',
 'Morocco':'Africa',
 'Pakistan':'Asia',
 'Tunisia':'Africa',
 'Ireland':'Europe',
 'Israel': 'Middle East',
 'Switzerland':'Europe',
 'Bangladesh':'Asia',
 'Romania':'Europe',
 'Austria':'Europe',
 'Belarus':'Europe',
 'Viet Nam':'Asia',
 'Czech Republic':'Europe',
 'Philippines':'Asia',
 'Belgium':'Europe',
 'New Zealand':'Oceania',
 'Norway':'Europe',
 'Finland':'Europe',
 'Egypt':'Africa',
 'Greece':'Europe',
 'Peru': 'South America',
 'Republic of Korea':'Asia'
}


# In[47]:


# Add continent to raw data dataframe
raw_data['Continent'] = raw_data['Q3'].replace(country_dict)


# In[48]:


raw_data['Continent'].unique()


# In[49]:


# Save a copy of the dataframe that contains text entries
raw_data_text = raw_data.copy()
raw_data_text.head(5)


# # 1.4 Encoding Categorical Data

# As previously mentioned, the dataframe above contains multiple categorical data which needs to be converted to text to be properly used with predictive models. 

# There are two main types of encoders that can be used: Label Encoder and One Hot Encoding. 
# - Label Encoding - will convert catrgorical data into numerical data representing each unique categorical answer. 
# - One Hot Encoding - will take a column which has been label encoded and split it into multiple columns and replacing each with 1s and 0s depending on what the column may have. 
# 
# One Hot Encoding is preferred for predictive methods as label encoding alone may confuse the model. Since there could be different numbers in the same column after label encoding, the model may potentially misunderstand the data to be in some kind of order or hierarchy (i.e. 0 < 1 <2). One can use one hot encoding to create a separate column to avoid this confusion. However, the drawback to this would be the increase in columns (i.e. features of the model). Both types of encoding were utilized in this assignment. 

# In this code, columns pertaining to age (Q2), continent of residence from feature engineering, education (Q4), undergraduate major (Q5) and job title (Q6) should be one hot encoded while the rest of the cateogorical data should be label encoded. The aforementioned questions to be one hot encoded should not be left label encoded as there may be an induced hierarchy/order as previously mentioned. 

# In[50]:


cleaned_encoded_data = raw_data_text.copy()


# In[51]:


cleaned_encoded_data.head(5)


# By glancing over potential answers in some of the critical and base questions (handpicked in the context of this assignment) Q2, Q4, Q5, Q6, Q12, and Q17, it is observed that some of the answers did not provide any beneficial information. Hence, answers such as 'Other', 'I prefer not to answer', 'I never declared a major' will be dropped from the data set. The changes made to cleaned_encoded_data will also be applied to raw_data_text. 

# In[52]:


print('Q2')
print(raw_data['Q2'].unique())
print('Q4')
print(raw_data['Q4'].unique())
print('Q5')
print(raw_data['Q5'].unique())
print('Q6')
print(raw_data['Q6'].unique())
print('Q12_MULTIPLE_CHOICE')
print(raw_data['Q12_MULTIPLE_CHOICE'].unique())
print('Q17')
print(raw_data['Q17'].unique())


# In[53]:


# Remove no information rows (i.e. "Other" entries) in Q2, Q5, Q6, Q12, Q17, Q4
cleaned_encoded_data = cleaned_encoded_data[raw_data['Q2'] != 'Other']
cleaned_encoded_data = cleaned_encoded_data[raw_data['Q4'] != 'I prefer not to answer']
cleaned_encoded_data = cleaned_encoded_data[raw_data['Q5'] != 'I never declared a major']
cleaned_encoded_data = cleaned_encoded_data[raw_data['Q5'] != 'Other']
cleaned_encoded_data = cleaned_encoded_data[raw_data['Q6'] != 'Other']
cleaned_encoded_data = cleaned_encoded_data[raw_data['Q12_MULTIPLE_CHOICE'] != 'Other']
cleaned_encoded_data = cleaned_encoded_data[raw_data['Q17'] != 'Other']
print(cleaned_encoded_data.shape)


# In[54]:


raw_data_text = cleaned_encoded_data.copy()


# In[55]:


# OneHotEncode Q2
one_hot_Q2 = pd.get_dummies(cleaned_encoded_data['Q2'])
cleaned_encoded_data = cleaned_encoded_data.drop('Q2', axis=1)
cleaned_encoded_data = cleaned_encoded_data.join(one_hot_Q2)


# In[56]:


# OneHotEncode continents
one_hot_Continent = pd.get_dummies(cleaned_encoded_data['Continent'])
cleaned_encoded_data = cleaned_encoded_data.drop('Continent', axis=1)
cleaned_encoded_data = cleaned_encoded_data.join(one_hot_Continent)


# In[57]:


# OneHotEncode Q5
one_hot_Q5 = pd.get_dummies(cleaned_encoded_data['Q5'])
cleaned_encoded_data = cleaned_encoded_data.drop('Q5', axis=1)
cleaned_encoded_data = cleaned_encoded_data.join(one_hot_Q5)


# In[58]:


# OneHotEncode Q6
one_hot_Q6 = pd.get_dummies(cleaned_encoded_data['Q6'])
cleaned_encoded_data = cleaned_encoded_data.drop('Q6', axis=1)
cleaned_encoded_data = cleaned_encoded_data.join(one_hot_Q6)


# In[59]:


cleaned_encoded_data.head(5)


# In[60]:


cleaned_encoded_data.dtypes


# In[61]:


# Encode categorical data using label encoder
encoder = LabelEncoder()
for column in cleaned_encoded_data.columns:
    if cleaned_encoded_data[column].dtypes == 'object':
        data = cleaned_encoded_data[column]
        encoder.fit(data.values)
        cleaned_encoded_data[column] = encoder.transform(cleaned_encoded_data[column])
cleaned_encoded_data = cleaned_encoded_data # update table


# In[62]:


cleaned_encoded_data.head(5)


# In[63]:


# Check if there's any column that contains purely zeros and remove it
cleaned_encoded_data = cleaned_encoded_data.loc[:, (cleaned_encoded_data != 0).any(axis=0)]
raw_data_text = raw_data_text.loc[:, (raw_data_text != 0).any(axis=0)]
len(cleaned_encoded_data.columns)


# From the cleaning, there are 276 columns remaining with the cleaned data set named cleaned_encoded_data. Furthermore, the text version of this data set prior to encoding is called raw_data_text. Two version of the data set are kept as the latter makes it easier to plot the dataframe for the exploratory analysis section of this notebook. 

# In[64]:


cleaned_encoded_data.head(5)


# # 3.0 Feature Engineering and Selection

# In this main section, we will look into engineering new features, performing another feature importance analysis to determine optimal features, and perform RFE (recursive feature elimination) using lasso regression (regularized regression) to determine features to be used in the ML models. 

# In[65]:


cleaned_encoded_data.head(5)


# ## 3.1 Feature Engineering

# ### Explain how feature engineering is a useful tool in machine learning. 
# Feature engineering is the process of transforming raw data into features that better represent the underlying problem to predictive models. The features in the data are important to the predictive models and influence the output of the model. Dealing with a large number of features, especially in the context of this project, is difficult due to high dimensionality. More features tends to make models more complex and difficult to interpret the data. Feature engineering allows us to create new features from the raw data presented that may better represent the data. This makes it easier to create models, train these models, and potentially achieve decent accuracies. 
# 
# In the following sections, we will be presenting features that were created from existing raw data set. These include:
# - Creation of a continent of residence based from country of residence
# - Combine multiple part questions from the raw data set
# - Target variable manipulation (square root transformation of the salary data to create a better distribution)
# - Scaling the features to have a mean of 0 and a standard deviation of 1.0 to easily model it in certain ML models such as SVM, etc. 
# 

# ### 3.1 a) Continent of Residence 

# This feature was already created and was encoded in section 1.0. 

# ### 3.1 b) Combine Part Questions

# Multiple question contains parts pertaining to specific tasks, programming languages, and cloud computing services used, etc. To reduce the number of features, the parts will be summed to account for the total # of programming languages known, cloud computing services used, etc. 
# 
# Questions that contain potentially relevant information: 
# - Q11_PartX: Select any activities that make up an important part of your role at work
# - Q12_PartX: Number of tools used to analyze data
# - Q13_PartX: Number of IDEs
# - Q14_PartX: Hosted Notebooks
# - Q15_PartX: Cloud computing services
# - Q16_PartX: Programming languages
# - Q19_PartX: ML Frameworks
# - Q21_PartX: DataVisualization
# - Q27_PartX: Cloud computing products
# - Q28_PartX: ML products
# - Q29_PartX: Relational database products used
# - Q30_PartX: Big Data and Analytic Products
# - Q31_PartX: Types of data interacted
# - Q33_PartX: Public Datasets used
# - Q36_PartX: Online platforms
# - Q38_PartX: Media sources
# - Q47_PartX: Preference for explaining decisions
# - Q48_Part9: Tools and methods to make your work easy to reproduce
# - Q50_PartX: Barriers

# In[66]:


# Copy dataframes
cleaned_encoded_data_edited = cleaned_encoded_data.copy()

# Additional features
Q11_list = []
Q12_list = []
Q13_list = []
Q14_list = []
Q15_list = []
Q16_list = []
Q19_list = []
Q21_list = []
Q27_list = []
Q28_list = []
Q29_list = []
Q30_list = []
Q31_list = []
Q33_list = []
Q36_list = []
Q38_list = []
Q47_list = []
Q48_list = []
Q50_list = []

cleaned_encoded_data_edited_list = list(cleaned_encoded_data_edited)

for column in cleaned_encoded_data_edited_list:
    column = column.rstrip()
    if "Q11_Part" in column:
        Q11_list.append(column)
    elif "Q12_Part" in column:
        Q12_list.append(column)
    elif "Q13_Part" in column:
        Q13_list.append(column)
    elif "Q14_Part" in column:
        Q14_list.append(column)
    elif "Q15_Part" in column:
        Q15_list.append(column) 
    elif "Q16_Part" in column:
        Q16_list.append(column)
    elif "Q19_Part" in column:
        Q19_list.append(column)
    elif "Q21_Part" in column:
        Q21_list.append(column)
    elif "Q27_Part" in column:
        Q27_list.append(column)
    elif "Q28_Part" in column:
        Q28_list.append(column)
    elif "Q29_Part" in column:
        Q29_list.append(column) 
    elif "Q30_Part" in column:
        Q30_list.append(column)
    elif "Q31_Part" in column:
        Q31_list.append(column)
    elif "Q33_Part" in column:
        Q33_list.append(column)
    elif "Q36_Part" in column:
        Q36_list.append(column)
    elif "Q38_Part" in column:
        Q38_list.append(column)
    elif "Q47_Part" in column:
        Q47_list.append(column) 
    elif "Q48_Part" in column:
        Q48_list.append(column)
    elif "Q50_Part" in column:
        Q50_list.append(column)
    else:
        continue


# In[67]:


# cleaned_encoded_data['Known Languages'] = cleaned_encoded_data[Q16_list].sum(axis=1)
cleaned_encoded_data_edited['Q11_sum'] = cleaned_encoded_data_edited[Q11_list].sum(axis=1)
cleaned_encoded_data_edited['Q12_sum'] = cleaned_encoded_data_edited[Q12_list].sum(axis=1)
cleaned_encoded_data_edited['Q13_sum'] = cleaned_encoded_data_edited[Q13_list].sum(axis=1)
cleaned_encoded_data_edited['Q14_sum'] = cleaned_encoded_data_edited[Q14_list].sum(axis=1)
cleaned_encoded_data_edited['Q15_sum'] = cleaned_encoded_data_edited[Q15_list].sum(axis=1)
cleaned_encoded_data_edited['Q16_sum'] = cleaned_encoded_data_edited[Q16_list].sum(axis=1)
cleaned_encoded_data_edited['Q19_sum'] = cleaned_encoded_data_edited[Q19_list].sum(axis=1)
cleaned_encoded_data_edited['Q21_sum'] = cleaned_encoded_data_edited[Q21_list].sum(axis=1)
cleaned_encoded_data_edited['Q27_sum'] = cleaned_encoded_data_edited[Q27_list].sum(axis=1)
cleaned_encoded_data_edited['Q28_sum'] = cleaned_encoded_data_edited[Q28_list].sum(axis=1)
cleaned_encoded_data_edited['Q29_sum'] = cleaned_encoded_data_edited[Q29_list].sum(axis=1)
cleaned_encoded_data_edited['Q30_sum'] = cleaned_encoded_data_edited[Q30_list].sum(axis=1)
cleaned_encoded_data_edited['Q31_sum'] = cleaned_encoded_data_edited[Q31_list].sum(axis=1)
cleaned_encoded_data_edited['Q33_sum'] = cleaned_encoded_data_edited[Q33_list].sum(axis=1)
cleaned_encoded_data_edited['Q36_sum'] = cleaned_encoded_data_edited[Q36_list].sum(axis=1)
cleaned_encoded_data_edited['Q38_sum'] = cleaned_encoded_data_edited[Q38_list].sum(axis=1)
cleaned_encoded_data_edited['Q47_sum'] = cleaned_encoded_data_edited[Q47_list].sum(axis=1)
cleaned_encoded_data_edited['Q48_sum'] = cleaned_encoded_data_edited[Q48_list].sum(axis=1)
cleaned_encoded_data_edited['Q50_sum'] = cleaned_encoded_data_edited[Q50_list].sum(axis=1)


# In[68]:


cleaned_encoded_data_edited.head(5)


# In[69]:


list(cleaned_encoded_data_edited.columns)


# In[70]:


colsToDrop = Q11_list + Q12_list + Q13_list + Q14_list + Q15_list + Q16_list + Q19_list + Q21_list + Q27_list + Q28_list + Q29_list + Q30_list + Q31_list + Q33_list + Q36_list + Q38_list + Q47_list + Q48_list + Q50_list 
# colsToDrop


# In[71]:


cleaned_encoded_data_edited = cleaned_encoded_data_edited.drop(colsToDrop, axis=1)


# In[72]:


cleaned_encoded_data_edited.head(5)


# ### 3.1 c) Target Variable Manipulation (Remove Outliers and SQRT Transform)

# In this section, we will look into the salaries found in the data set. Based on a histogram plot of the salaries, we can see that it is skewed with a right tail. Based on visual observation, we can see that there are potential outliers found for those individuals who earn more than USD350,000. While this may be a realistic value for certain senior individuals, this is not typically the case for an average data scientist. Hence, we will be removing individuals who earn more than USD350,000. 
# 
# Afterwards, we will be performing a square root transformation to the salaries data to create a better distribution closer to a normal distribution as it will provide better predictions and estimates and thus, achieve better accuracies overall. 

# In[73]:


plt.hist(cleaned_encoded_data_edited['Q9'], edgecolor='black')
plt.title('Salary Distribution')
plt.ylabel('count')
plt.xlabel('Salary (USD$)')
plt.show()


# In[74]:


# print how many individuals earned more than USD$350,000
len(cleaned_encoded_data_edited[cleaned_encoded_data_edited['Q9'] > 350000])


# In[75]:


# remove outliers
cleaned_encoded_data_edited=cleaned_encoded_data_edited[cleaned_encoded_data_edited['Q9'] < 350000]


# In[76]:


# Perform Square Root transformation
Salary_Y = np.sqrt(cleaned_encoded_data_edited['Q9'])


# In[77]:


plt.hist(Salary_Y, edgecolor='black')
plt.title('Salary Distribution After Transformation')
plt.ylabel('count')
plt.xlabel('Salary (USD$)')
plt.show()


# ### 3.1 d) Scaling Features

# Prior to using any ML model, the data needs to be scaled. Each column will be scaled to have a mean of zero and a standard deviation of one. This is to ensure that all features and data used are within the same order of magnitude. Providing data with different scales will skew the data and thus, affect the performance of the models. 

# In[78]:


# Drop Q9 - Salary information as it does not need to be scaled. 

cleaned_encoded_data_edited = cleaned_encoded_data_edited.drop('Q9', axis=1)


# In[79]:


# use standardscaler to scale all the features
scaled_features = StandardScaler().fit_transform(cleaned_encoded_data_edited.values)
scaled_features_df = pd.DataFrame(scaled_features, index=cleaned_encoded_data_edited.index, columns = cleaned_encoded_data_edited.columns)


# In[80]:


scaled_features_df.head(5)


# ## 3.2 Feature Selection

# In this section, we will discuss methods to select optimal features to be used in our models. We will first define a function that was modified from tutorial to perform kfold cross validation for the training sets. After, we will use RFE using linear regression, followed by RFE using lasso regression to see any changes/improvements. Upon selected features, another feature importance analysis is performed similar to section 2.2.    

# In[81]:


def run_kfold(X, Y, model):
    
    kf = KFold(n_splits=10) #n_splits previously n_folds
    
    outcomes = []
    fold = 0
    
    for train_index, test_index in kf.split(X):
        fold += 1
        X_train, X_test = X.values[train_index], X.values[test_index]
        Y_train, Y_test = Y.values[train_index], Y.values[test_index]
        
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        
        accuracy = r2_score(Y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))   
        
    mean_outcome = np.mean(outcomes)
    std_outcome=np.std(outcomes)
    var_outcome = np.var(outcomes)
    print("Mean r2: {0}".format(mean_outcome)) 
    print("Standard Deviation: {0}".format(std_outcome)) 
    print("Variance: {0}".format(var_outcome)) 
    
    return {'mean': mean_outcome, 'std': std_outcome, 'var': var_outcome, 'outcomes': outcomes}


# In[82]:


# Define inputs and output of the models 
Salaries_Y = Salary_Y # transformed salaries
Salaries_X = scaled_features_df # scaled dataframe


# In[83]:


Salaries_X.head(5)


# Split the data between training and testing sets with a test size of 0.3. 

# In[84]:


# Train Test Split
Salaries_X_train, Salaries_X_test, Salaries_Y_train, Salaries_Y_test = train_test_split(Salaries_X,Salaries_Y,test_size=0.3, random_state=101)


# Let's fit a quick linear regression model to the data. 

# In[85]:


model_lr = LinearRegression()
#run_kfold(Salaries_X, Salaries_Y, model_lr)
model_lr.fit(Salaries_X_train, Salaries_Y_train)
predictions = model_lr.predict(Salaries_X_test)
Test_Score = r2_score(Salaries_Y_test, predictions)

print ("------------------\nTest Score: " + str(Test_Score))

model_lr.fit(Salaries_X_train, Salaries_Y_train)


# In[86]:


mean_absolute_error(Salaries_Y_test,predictions)


# In[87]:


print(sorted(list(zip(model_lr.coef_, Salaries_X)))[0:10])


# We can see that the coefficients in front of certain features are large. We will use RFE to identify high ranking features. 

# In[88]:


rfe = RFE(model_lr)
fit = rfe.fit(Salaries_X_train, Salaries_Y_train) 


# In[89]:


# sorted(list(zip(fit.ranking_,Salaries_X))[0:10])
sorted(list(zip(fit.ranking_,Salaries_X)))


# Using a linear regression model, the goal of recursive feature elimination is to select features by recursively considering smaller and smaller sets of features. The estimator is first trained on an initial set of features and the importance of each feature is obtained through a coef_ attribute or through a feature_importances_attribute. The least important features are removed from the current set and this process is repeated until a desired number of features to select is reached or the algorithm yielded no more improvement. 

# From the list above, we can see that age, location, job title, gender, and a set skills of programming languages/tools are among those features that are ranked high. We will use regularized regression as well to see if it yielded similar results. 

# ### 3.2.1 Feature Selection: Lasso Regularization

# In this subsection, we will use lasso along with RFE to select features. We will first fit the data using lasso regression, and then use RFE's ranking system to determine ideal features. 

# In[90]:


reg = Lasso(alpha = 0.5,max_iter=10000)
reg.fit(Salaries_X_train, Salaries_Y_train)
reg.score(Salaries_X_train, Salaries_Y_train)


# In[91]:


reg_gridsearch = Lasso(random_state=125)
#Parameters to test
parameters = {'alpha':[0.2, 0.5,1,3,5, 10], # Constant that multiplies the L1 term. Defaults to 1.0.
             'normalize':[True,False]} #

# Compare parameters by score of model 
acc_scorer_lm = make_scorer(r2_score)

# Run the grid search
grid_obj_lm = GridSearchCV(reg_gridsearch, parameters, scoring=acc_scorer_lm)
grid_obj_lm = grid_obj_lm.fit(Salaries_X_train, Salaries_Y_train)

reg_gridsearch = grid_obj_lm.best_estimator_  #Select best parameter combination


# In[92]:


reg_gridsearch # print out the optimal params so grid search does not need to be rerun


# In[93]:


print('alpha (Constant that multiplies the L1 term):',grid_obj_lm.best_estimator_.alpha) 
print('normalize:',grid_obj_lm.best_estimator_.normalize)


# In[94]:


predictions_lasso=reg_gridsearch.predict(Salaries_X_test)


# In[95]:


mean_absolute_error(Salaries_Y_test,predictions_lasso)


# In[96]:


result_lasso_fs = run_kfold(Salaries_X_train, Salaries_Y_train, reg_gridsearch)
Test_Score = r2_score(Salaries_Y_test, predictions_lasso)

print ("------------------\nTest Score: " + str(Test_Score))


# In[97]:


# result_lasso_fs['mean']
# result_lasso_fs['std']
# result_lasso_fs['var']


# In[98]:


sorted(list(zip(reg.coef_, Salaries_X)),reverse=True)[0:10]


# In[99]:


rfe_lasso = RFE(reg)
fit_lasso = rfe_lasso.fit(Salaries_X_train, Salaries_Y_train) #Sampling because of slow run time


# In[100]:


sorted(list(zip(fit_lasso.ranking_,Salaries_X)))


# Similar to RFE using linear regression, lasso regression yielded the same feature ranking as with RFE using linear regression. Hence, the features selected for this project will be based on the ranking system produced by the RFE algorithm from the lasso regression. 

# ## 3.3 Modifying Dataset with Selected Features

# Based on the selected features, those features that were ranked 1 were kept in the dataset. All other columns were dropped as they did not rank high from the aforementioned section. 

# In[101]:


selected_features = sorted(list(zip(fit_lasso.ranking_,Salaries_X_train)))


# In[102]:


len(selected_features)


# In[103]:


# Select only the features that were ranked the highest
ranking = []
selected_columns = []
for i in selected_features:
    if i[0] < 2:
        ranking.append(i[0])
        selected_columns.append(i[1])
    else:
        continue


# In[104]:


len(ranking)


# In[105]:


# create subset of dataframe with selected columns
# Subset data is the dataframe that contains selected columns from the feature selection algorithm
subset_data = Salaries_X.copy()
subset_data = subset_data[selected_columns]


# In[106]:


# include Q9 back into main data to get correlation matrix
subset_data['Salary'] = Salaries_Y


# In[107]:


subset_data.head()
print(len(subset_data))


# ### 3.3.1 Feature Importance of Selected Features

# We will again look into the correlation matrix to visualize feature importance among the selected features. 

# In[108]:


corr_subset = subset_data.corr()


# In[109]:


sns.set(style="white")
mask = np.zeros_like(corr_subset, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_subset, mask=mask, cmap=cmap, vmax=.5, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Let's look into the specific correlations between the selected features and the salary. 

# In[110]:


corr_subset_Salary = corr_subset['Salary']
indices_subset = corr_subset['Salary'].index


# In[111]:


features_edited = []
value_corr_feature_edited = []

for value, index in zip(corr_subset_Salary, indices_subset):
    if abs(value) > 0.2:
        features_edited.append(index)
        value_corr_feature_edited.append(value)


# In[112]:


features_edited


# In[113]:


value_corr_feature_edited


# In[114]:


plt.figure(figsize=(10,6))
ax=sns.barplot(x=features_edited, y=value_corr_feature_edited)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="center")
plt.title('Correlations with Respect to Salary of Selected Variables After RFE')
plt.xlabel('Features')
plt.ylabel('Correlation to Salary')
plt.show()


# 
#  'Q27_sum',
#  'Q29_sum',
#  'Q42_Part_1',
#  'Q47_sum',
#  'Q8',
#  'Student',
#  'Salary'
# 
# Features with the highest correlation: 
# - Age group ('18-21', '22-24') are negatively correlated with Salary - makes sense since the younger you are, the less experience you have. Hence, your pay will be less than someone who has more job experience. 
# - Q3 (Country) and Continent ('Asia', 'North America') - data scientists working in North America earn more than those in Asia
# - Q10 - Do you use ML at work? - if ML is being used at work, then it is positively correlated to salary
# - Q11 - Number of tasks done in their respective job - the more responsibilities and roles one has, the higher the pay
# - Q27 - Number of cloud computing products used in the last 5 years - the more experience with a variety of cloud computing tools used, the higher the salary
# - Q30 - Number of big data and analytics products used in the last 5 years - the more experience with a variety of big data and analytics products used, the higher the salary
# - Q29 - Number of database products used 
# - Q42_Part_1 - If the models developed are related to revenue of the business, then the individual will be paid more
# - Q47_sum - Preference for explaining and intrepreting ML models
# - Student - If the individual is a student, then they will be paid less. 

# In[115]:


# Assign Salaries_x to the new subset data containing selected features 
Salaries_Y = subset_data['Salary']
Salaries_X = subset_data.drop(['Salary'], axis=1)

# Train Test Split
Salaries_X_train, Salaries_X_test, Salaries_Y_train, Salaries_Y_test = train_test_split(Salaries_X,Salaries_Y,test_size=0.3, random_state=101)


# In[116]:


print(len(Salaries_X)) # greater than 5000 points so it's good! 


# ## 3.0 Summary
# In this section, new features were created from existing features from the raw data set. A feature selection algorithm, RFE applied on a lasso regularized regression model, was applied to select the ideal features among the newly created ones and among the existing features from the data set. Features were ranked based on their performance on the RFE algorithm and those that ranked the highest were chosen to be used for implementation with the model. 
# 
# Feature dimensionality reduction such as PCA was not used as it yielded lower accuracies than keeping all the features in the model. PCA will aid in helping make the data set easier to model by reducing its dimensionality at the expense of explainable variance. Hence, PCA was not chosen for this application. 

# # 4.0 Models Selected from Assignment

# In this section, we will use the optimal models determined via gridsearch from the main body of the assignment. Please see the original assignment python notebook for more details. 

# ## 4.1 Lasso Linear Regression

# In[117]:


# Final Result of GridSearchCV
optimal_lasso = Lasso(alpha=0.0001, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)

result_model_lasso_tuned = run_kfold(Salaries_X_train, Salaries_Y_train, optimal_lasso)
predictions_lasso_tuned = optimal_lasso.predict(Salaries_X_test)
Test_Score_lasso_tuned = r2_score(Salaries_Y_test, predictions_lasso_tuned)

print("Test Score: ")
print(Test_Score_lasso_tuned)


# ## 4.2 Gradient Boosting Regression

# In[118]:


# Final Result of GridSearchCV
optimal_GB = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=250, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)

result_model_GB_tuned = run_kfold(Salaries_X_train, Salaries_Y_train, optimal_GB)
predictions_GB_tuned = optimal_GB.predict(Salaries_X_test)
Test_Score_GB_tuned = r2_score(Salaries_Y_test, predictions_GB_tuned)

print("Test Score: ")
print(Test_Score_GB_tuned)


# ## 4.3 Random Forest Regressor

# In[119]:


# Final Result of GridSearchCV
optimal_RF = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=50,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

result_model_RF_tuned = run_kfold(Salaries_X_train, Salaries_Y_train, optimal_RF)
predictions_RF_tuned = optimal_RF.predict(Salaries_X_test)
Test_Score_RF_tuned = r2_score(Salaries_Y_test, predictions_RF_tuned)

print("Test Score: ")
print(Test_Score_RF_tuned)


# ## 4.4 Support Vector Regressor

# In[120]:


# Final Result of GridSearchCV
optimal_SVR = SVR(C=0.1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
  kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

result_model_SVR_tuned = run_kfold(Salaries_X_train, Salaries_Y_train, optimal_SVR)
predictions_SVR_tuned = optimal_SVR.predict(Salaries_X_test)
Test_Score_SVR_tuned = r2_score(Salaries_Y_test, predictions_SVR_tuned)

print("Test Score: ")
print(Test_Score_SVR_tuned)


# ## 4.5 Combine Metrics to Compare With Neural Network

# In[121]:


predictions_lasso_tuned_test = optimal_lasso.predict(Salaries_X_test)
predictions_GB_tuned_test = optimal_GB.predict(Salaries_X_test)
predictions_RF_tuned_test = optimal_RF.predict(Salaries_X_test)
predictions_SVR_tuned_test = optimal_SVR.predict(Salaries_X_test)

# Coefficient of Determination
R2_Lasso = r2_score(Salaries_Y_test, predictions_lasso_tuned_test)
R2_GB = r2_score(Salaries_Y_test, predictions_GB_tuned_test)
R2_RF = r2_score(Salaries_Y_test, predictions_RF_tuned_test)
R2_SVR = r2_score(Salaries_Y_test, predictions_SVR_tuned_test)

# MSE - mean squared error
MSE_Lasso = mean_squared_error(Salaries_Y_test, predictions_lasso_tuned_test)
MSE_GB = mean_squared_error(Salaries_Y_test, predictions_GB_tuned_test)
MSE_RF = mean_squared_error(Salaries_Y_test, predictions_RF_tuned_test)
MSE_SVR = mean_squared_error(Salaries_Y_test, predictions_SVR_tuned_test)

#RMSE - root mean squared error
RMSE_Lasso = np.sqrt(MSE_Lasso)
RMSE_GB = np.sqrt(MSE_GB)
RMSE_RF = np.sqrt(MSE_RF)
RMSE_SVR = np.sqrt(MSE_SVR)


# # 5.0 Neural Network Implementation - BONUS

# In this section, we will implement a neural network to predict the target variable, salary. We will first experiment with different neural network architectures such as the number of hidden layers and nodes per layer, learning rate, number of iterations, and momentum. The parametric study of the neural network will first use the solver sgd - stochastic gradient descent - as most of the parameters are only valid with this solver option.  
# 
# GridSearchCV segments in this section are commented out as per the instructions on the assignment. 

# ## 5.1 Parametric Study - Number of Hidden Layers and Nodes per layer

# In this study, we will use a multi-layer perceptron regressor by changing the structure of the neural network and by keeping certain parameters constant. 

# In[122]:


model_NN_hidden_layers =  MLPRegressor(solver='sgd', alpha=1e-5, random_state=1, max_iter=1000)
model_NN_hidden_layers


# In[126]:


# model_NN_hidden_layers_param = {
#     "hidden_layer_sizes": [(1), # one layer but one neuron
#                            (2),
#                            (4),
#                            (8),
#                            (10),
#                            (1,2),
#                            (2,1),
#                            (2,4), # two layer with two neurons in the first, 4 in the second
#                            (1,2,3),
#                            (2,4,6),
#                            (3,2)]
# }

# # cv of 5 was chosen to reduce computational time
# cv_model_NN_hidden_layers = GridSearchCV(model_NN_hidden_layers, cv = 5, param_grid = model_NN_hidden_layers_param, n_jobs=-1, verbose = 100)
# cv_model_NN_hidden_layers.fit(Salaries_X_train, Salaries_Y_train)

# print('Optimal parameters found: ')
# print(cv_model_NN_hidden_layers.best_params_)
# print(cv_model_NN_hidden_layers.best_estimator_)


# In[123]:


# optimal_NN_hidden_layers = cv_model_NN_hidden_layers.best_estimator_

# predictions_NN_hidden_layers = optimal_NN_hidden_layers.predict(Salaries_X_test)
# Test_Score_NN_hidden_layers = r2_score(Salaries_Y_test, predictions_NN_hidden_layers)

# print("Test Score: ")
# print(Test_Score_NN_hidden_layers)


# In[124]:


# # Results for training set
# means_NN_hidden_layers = cv_model_NN_hidden_layers.cv_results_['mean_test_score']
# stds_NN_hidden_layers = cv_model_NN_hidden_layers.cv_results_['std_test_score']

# # Print all training cases
# for mean, std, params in zip(means_NN_hidden_layers, stds_NN_hidden_layers, cv_model_NN_hidden_layers.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))


# Based on the above performance from changing the structure, we can notice that the performance decreases significantly with deeper layers. This is most likely attributed to the vanishing gradient problem. Each of the nerual network;s weights receive an update proportional to the gradient of the error, with respect to the current weight in each iteration of training. The problem with different layers arises from when the gradient could be decraeasing and diminishing into a very small rate, effectively preventing the weight from changing value. Thus, this may inhibit the training procedure of the neural network, resulting in poor performance. 
# 
# The vanishing gradient problem is not evident once a smaller hidden layer size is used. Interestingly, a single layer will be suitable for the data set from the assignment. This is not surprising as the performance for linear regression from the previous sections (and from the main assignment) was decent, indicating that the data set may be linearly separable. As a result, we may not require a lot of nodes to capture this data set. Noticeably, the performance also changes as more nodes are added to the single hidden layer. However, there seems to be trade-off that occurs as implementing more nodes seems to have also decreased the network's performance. This may be due to the additional weight and computation that may be caused from excessive nodes in the neural net. 
# 
# The result of this grid search from varying the structure of the network indicated that a hidden layer size of 1 layer and 4 nodes is sufficient. 

# In[123]:


# Final Result of GridSearchCV
optimal_hidden_layers = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=4, learning_rate='constant',
       learning_rate_init=0.001, max_iter=1000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

result_model_hidden_layers = run_kfold(Salaries_X_train, Salaries_Y_train, optimal_hidden_layers)
predictions_hidden_layers = optimal_hidden_layers.predict(Salaries_X_test)
Test_Score_hidden_layers = r2_score(Salaries_Y_test, predictions_hidden_layers)

print("Test Score: ")
print(Test_Score_hidden_layers)


# ## 5.2 Parametric Study - Momentum

# In this study, we will use a multi-layer perceptron regressor by changing the momentum of the neural network and by keeping certain parameters constant. The momentum variable is used for the gradient descent update, used in the solver for the neural network regressor. 

# In[124]:


model_NN_momentum =  MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(4), random_state=1, max_iter=1000)
model_NN_momentum


# In[133]:


# model_NN_momentum_param = {
#     "momentum": [0.9, 0.8, 0.75, 0.5, 0.25]
# }

# # cv of 5 was chosen to reduce computational time
# cv_model_NN_momentum = GridSearchCV(model_NN_momentum, cv = 5, param_grid = model_NN_momentum_param, n_jobs=-1, verbose = 100)
# cv_model_NN_momentum.fit(Salaries_X_train, Salaries_Y_train)

# print('Optimal parameters found: ')
# print(cv_model_NN_momentum.best_params_)
# print(cv_model_NN_momentum.best_estimator_)


# In[135]:


# optimal_NN_momentum = cv_model_NN_momentum.best_estimator_

# predictions_NN_momentum = optimal_NN_momentum.predict(Salaries_X_test)
# Test_Score_NN_momentum = r2_score(Salaries_Y_test, predictions_NN_momentum)

# print("Test Score: ")
# print(Test_Score_NN_momentum)


# In[136]:


# # Results for training set
# means_NN_momentum = cv_model_NN_momentum.cv_results_['mean_test_score']
# stds_NN_momentum = cv_model_NN_momentum.cv_results_['std_test_score']

# # Print all training cases
# for mean, std, params in zip(means_NN_momentum, stds_NN_momentum, cv_model_NN_momentum.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))


# Based on the gridsearch results above, the optimal momentum parameter value is 0.5. The momentum is used to guide the gradient descent algorithm to a local minimum particularly in its application on the gradient of the error or loss function. The momentum value is analogous to the step size in many numerical methods, which is a factor applied to the heading (step or direction) of the function. A higher value may allow for a faster movement, particularly when multiplied by the gradient, whereas smaller values represent small steps in the right direction. In this case, the momentum value is 0.5. 

# In[125]:


# Final Result of GridSearchCV
optimal_momentum = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=4, learning_rate='constant',
       learning_rate_init=0.001, max_iter=1000, momentum=0.5,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

result_model_momentum = run_kfold(Salaries_X_train, Salaries_Y_train, optimal_momentum)
predictions_momentum = optimal_momentum.predict(Salaries_X_test)
Test_Score_momentum = r2_score(Salaries_Y_test, predictions_momentum)

print("Test Score: ")
print(Test_Score_momentum)


# ## 5.3 Parametric Study - Learning Rate

# In this study, we will use a multi-layer perceptron regressor by changing the learning rate of the neural network and by keeping certain parameters constant. The learning rate parameter controls the schedule for weight updates between the regressors. There are three main options for the learning rate
# - constant learning rate - controlled by learning_rate_init
# - invscaling - decreases learning rate at each time t using a function for the effective learning rate
# - adaptive - keeps changing the learning rate as long as the training loss keeps decreasing. 

# In[126]:


model_NN_learning_rate =  MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(4), momentum = 0.5,random_state=1, max_iter=1000)
model_NN_learning_rate


# In[126]:


# model_NN_learning_rate_param = {
#     "learning_rate": ['constant', 'invscaling', 'adaptive']
# }

# # cv of 5 was chosen to reduce computational time
# cv_model_NN_learning_rate = GridSearchCV(model_NN_learning_rate, cv = 5, param_grid = model_NN_learning_rate_param, n_jobs=-1, verbose = 100)
# cv_model_NN_learning_rate.fit(Salaries_X_train, Salaries_Y_train)

# print('Optimal parameters found: ')
# print(cv_model_NN_learning_rate.best_params_)
# print(cv_model_NN_learning_rate.best_estimator_)


# In[127]:


# optimal_NN_learning_rate = cv_model_NN_learning_rate.best_estimator_

# predictions_NN_learning_rate = optimal_NN_learning_rate.predict(Salaries_X_test)
# Test_Score_NN_learning_rate = r2_score(Salaries_Y_test, predictions_NN_learning_rate)

# print("Test Score: ")
# print(Test_Score_NN_learning_rate)


# In[128]:


# # Results for training set
# means_NN_learning_rate = cv_model_NN_learning_rate.cv_results_['mean_test_score']
# stds_NN_learning_rate = cv_model_NN_learning_rate.cv_results_['std_test_score']

# # Print all training cases
# for mean, std, params in zip(means_NN_learning_rate, stds_NN_learning_rate, cv_model_NN_learning_rate.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))


# Based from the grid search on the learning rate, it was found that the adaptive learning rate was the most optimal as it achieved the highest accuracies among the available options. The constant learning rate parameter utilized the defaulted learning rate set forth by sklearn. Noticeably, the invscaling performed the worst as it decreased the learning rate over time. This may result in a slower convergence to the optimal model. The adaptive model was able to change the learning rate over time to meet the minimum of the model's loss function. Hence, using the adaptive setting would allow for more flexible usages of the neural network instead of tinkering with specific learning rate init values for the constant learning rate parameter. 

# In[127]:


# Final Result of GridSearchCV
optimal_learning_rate = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=4, learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=1000, momentum=0.5,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

result_model_learning_rate = run_kfold(Salaries_X_train, Salaries_Y_train, optimal_learning_rate)
predictions_learning_rate = optimal_momentum.predict(Salaries_X_test)
Test_Score_learning_rate = r2_score(Salaries_Y_test, predictions_learning_rate)

print("Test Score: ")
print(Test_Score_learning_rate)


# # 5.4 Parametric Study - Number of Iterations

# In this study, we will use a multi-layer perceptron regressor by changing the number of maximum iterations of the neural network. The solver iteratures until convergence is defined based on a tolerance value or by the number of iterations. 

# In[130]:


model_NN_iterations =  MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(4), momentum = 0.5,learning_rate = 'adaptive', random_state=1)
model_NN_iterations


# In[131]:


# model_NN_iterations_param = {
#     "max_iter": [500, 750, 1000, 1500, 2000]
# }

# # cv of 5 was chosen to reduce computational time
# cv_model_NN_iterations = GridSearchCV(model_NN_iterations, cv = 5, param_grid = model_NN_iterations_param, n_jobs=-1, verbose = 100)
# cv_model_NN_iterations.fit(Salaries_X_train, Salaries_Y_train)

# print('Optimal parameters found: ')
# print(cv_model_NN_iterations.best_params_)
# print(cv_model_NN_iterations.best_estimator_)


# In[132]:


# optimal_NN_iterations = cv_model_NN_iterations.best_estimator_

# predictions_NN_iterations = optimal_NN_iterations.predict(Salaries_X_test)
# Test_Score_NN_iterations = r2_score(Salaries_Y_test, predictions_NN_iterations)

# print("Test Score: ")
# print(Test_Score_NN_iterations)


# In[133]:


# # Results for training set
# means_NN_iterations = cv_model_NN_iterations.cv_results_['mean_test_score']
# stds_NN_iterations = cv_model_NN_iterations.cv_results_['std_test_score']

# # Print all training cases
# for mean, std, params in zip(means_NN_iterations, stds_NN_iterations, cv_model_NN_iterations.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))


# From the results of the gridsearch by varying the maximum number of iterations, it was found that it didn't change. This indicates that the tolerance level to satisfy convergence must have been met prior to the maximum iteration constraint. Hence, to reduce computational time, the lowest max_iter was selected. 

# In[132]:


# Final Result of GridSearchCV
optimal_iterations = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=4, learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=500, momentum=0.5,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

result_model_iterations = run_kfold(Salaries_X_train, Salaries_Y_train, optimal_iterations)
predictions_iterations = optimal_iterations.predict(Salaries_X_test)
Test_Score_iteration = r2_score(Salaries_Y_test, predictions_iterations)

print("Test Score: ")
print(Test_Score_iteration)


# ## 5.4 Optimal Neural Network

# In[133]:


model_NN_solver =  MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(4), momentum = 0.5,learning_rate = 'adaptive', random_state=1, max_iter=500)
model_NN_solver


# In[151]:


# model_NN_solver_param = {
#     'hidden_layer_sizes': [(1), (2), (2,4), (5,2), (10, 5, 8)],
#      "momentum": [0.9, 0.8, 0.75, 0.5, 0.25],
#      "learning_rate": ['constant', 'invscaling', 'adaptive']
#     }


# # cv of 5 was chosen to reduce computational time
# cv_model_NN_solver = GridSearchCV(model_NN_solver, cv = 5, param_grid = model_NN_solver_param, n_jobs=-1, verbose = 100)
# cv_model_NN_solver.fit(Salaries_X_train, Salaries_Y_train)

# print('Optimal parameters found: ')
# print(cv_model_NN_solver.best_params_)
# print(cv_model_NN_solver.best_estimator_)


# In[152]:


# optimal_NN_solver = cv_model_NN_solver.best_estimator_

# predictions_NN_solver = optimal_NN_iterations.predict(Salaries_X_test)
# Test_Score_NN_solver = r2_score(Salaries_Y_test, predictions_NN_solver)

# print("Test Score: ")
# print(Test_Score_NN_solver)


# In[153]:


# # Results for training set
# means_NN_solver = cv_model_NN_iterations.cv_results_['mean_test_score']
# stds_NN_iterations = cv_model_NN_iterations.cv_results_['std_test_score']

# # Print all training cases
# for mean, std, params in zip(means_NN_solver, stds_NN_iterations, cv_model_NN_solver.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))


# In[134]:


# Final Result of GridSearchCV
optimal_NN = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=2, learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=500, momentum=0.5,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

result_model_solver = run_kfold(Salaries_X_train, Salaries_Y_train, optimal_NN)
predictions_solver = optimal_NN.predict(Salaries_X_test)
Test_Score_solver = r2_score(Salaries_Y_test, predictions_solver)

print("Test Score: ")
print(Test_Score_solver)


# In this section, we performed a gridsearch by varying the hyperparameters involving the structure of the neural network. It was found that optimal nerual network had one hidden layer with two nodes, adaptive learning, and a momentum of 0.5. This is not surprising as these parameters arose from the individual parametric study performed on the structure of the neural network. 

# # 6.0 Comparison to Other Models

# In this section, we will plot performance metrics such as $R^{2}$ and RMSE of the NN with respect to the other models presented. 

# In[135]:


predictions_NN_test = optimal_NN.predict(Salaries_X_test)
R2_NN = r2_score(Salaries_Y_test, predictions_NN_test)
MSE_NN = mean_squared_error(Salaries_Y_test, predictions_NN_test)
RMSE_NN = np.sqrt(MSE_NN)


# In[136]:


# define lists to aid with plotting
model = ['Lasso', 'GB', 'RF', 'SVR', 'NN']
R2_test = [R2_Lasso, R2_GB, R2_RF, R2_SVR, R2_NN]
RMSE_test = [RMSE_Lasso, RMSE_GB, RMSE_RF, RMSE_SVR, RMSE_NN]


# In[137]:


# Plot R2
plt.figure(figsize=(7,5))
ax=sns.barplot(x=model, y=R2_test, palette="rocket")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="center")
plt.title('Test Scores After Tuning ($R^{2}$)')
plt.xlabel('Models')
plt.ylabel('$R^{2}$')
plt.ylim(0, 0.75)
plt.show()


# In[138]:


R2_test


# In[139]:


# Plot RMSE
plt.figure(figsize=(7,5))
ax=sns.barplot(x=model, y=RMSE_test, palette="rocket")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="center")
plt.title('RMSE of Models After Tuning')
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.show()


# In[140]:


RMSE_test


# Based on the above plots of R2 and RMSE among all the tuned models, the best performing model is still the Gradient Boosting Regressor as it achieved the highest coefficient of determination R2 and the smallest RMSE. 
# 
# Hence, the models were ranked based on their test R2 score and RMSE in descending order:
# - Gradient Boosting - BEST
# - Random Forest
# - Multi-layer perceptron Neural Network using Stochastic Gradient Descent
# - SVR
# - Lasso Linear Regression
# 
# #### Discuss pros and cons of using a neural network for this data set
# 
# Pros: 
# - Easy to implement
# - Ability to learn and model non-linear models and complex relationships that may be evident in the provided data set
# 
# Cons:
# - Lack of explainability (acts like a blackbox) unlike other models available
# - Multi-layer neural networks are really hard to train and require tuning a lot of its parameters
# - Creating and tuning a neural network is computationally expensive. There may exist other traditional ML methods that can better capture and represent the data. 
# 

# #### Conclusion: 
# Based on the comparisons between the models, it is apparent that while a neural network can be easily applied to any ML problem, it is occasionally not the ideal model to select as it is dependent on the data set. Typically, neural networks perform better for larger data sets. As the data set used after cleaning was only around ~8000 points, it was not sufficient for the neural network to learn the intricate relations between the features in the dataset. Moreover, there exists easier and traditional models to represent and model the data such as GB or RF that are easily explainable and understandable. 
# 
