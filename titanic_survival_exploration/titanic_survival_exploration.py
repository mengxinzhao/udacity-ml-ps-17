
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Introduction and Foundations
# ## Project: Titanic Survival Exploration
# 
# In 1912, the ship RMS Titanic struck an iceberg on its maiden voyage and sank, resulting in the deaths of most of its passengers and crew. In this introductory project, we will explore a subset of the RMS Titanic passenger manifest to determine which features best predict whether someone survived or did not survive. To complete this project, you will need to implement several conditional predictions and answer the questions below. Your project submission will be evaluated based on the completion of the code and your responses to the questions.
# > **Tip:** Quoted sections like this will provide helpful instructions on how to navigate and use an iPython notebook. 

# # Getting Started
# To begin working with the RMS Titanic passenger data, we'll first need to `import` the functionality we need, and load our data into a `pandas` DataFrame.  
# Run the code cell below to load our data and display the first few entries (passengers) for examination using the `.head()` function.
# > **Tip:** You can run a code cell by clicking on the cell and using the keyboard shortcut **Shift + Enter** or **Shift + Return**. Alternatively, a code cell can be executed using the **Play** button in the hotbar after selecting it. Markdown cells (text cells like this one) can be edited by double-clicking, and saved using these same shortcuts. [Markdown](http://daringfireball.net/projects/markdown/syntax) allows you to write easy-to-read plain text that can be converted to HTML.

# In[1]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic('matplotlib inline')

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# Print the first few entries of the RMS Titanic data
display(full_data.head())


# From a sample of the RMS Titanic data, we can see the various features present for each passenger on the ship:
# - **Survived**: Outcome of survival (0 = No; 1 = Yes)
# - **Pclass**: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
# - **Name**: Name of passenger
# - **Sex**: Sex of the passenger
# - **Age**: Age of the passenger (Some entries contain `NaN`)
# - **SibSp**: Number of siblings and spouses of the passenger aboard
# - **Parch**: Number of parents and children of the passenger aboard
# - **Ticket**: Ticket number of the passenger
# - **Fare**: Fare paid by the passenger
# - **Cabin** Cabin number of the passenger (Some entries contain `NaN`)
# - **Embarked**: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)
# 
# Since we're interested in the outcome of survival for each passenger or crew member, we can remove the **Survived** feature from this dataset and store it as its own separate variable `outcomes`. We will use these outcomes as our prediction targets.  
# Run the code cell below to remove **Survived** as a feature of the dataset and store it in `outcomes`.

# In[2]:


# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
display(data.head())


# The very same sample of the RMS Titanic data now shows the **Survived** feature removed from the DataFrame. Note that `data` (the passenger data) and `outcomes` (the outcomes of survival) are now *paired*. That means for any passenger `data.loc[i]`, they have the survival outcome `outcomes[i]`.
# 
# To measure the performance of our predictions, we need a metric to score our predictions against the true outcomes of survival. Since we are interested in how *accurate* our predictions are, we will calculate the proportion of passengers where our prediction of their survival is correct. Run the code cell below to create our `accuracy_score` function and test a prediction on the first five passengers.  
# 
# **Think:** *Out of the first five passengers, if we predict that all of them survived, what would you expect the accuracy of our predictions to be?*

# In[3]:


def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
# Test the 'accuracy_score' function
predictions = pd.Series(np.ones(5, dtype = int))
print(accuracy_score(outcomes[:5], predictions))


# > **Tip:** If you save an iPython Notebook, the output from running code blocks will also be saved. However, the state of your workspace will be reset once a new session is started. Make sure that you run all of the code blocks from your previous session to reestablish variables and functions before picking up where you last left off.
# 
# # Making Predictions
# 
# If we were asked to make a prediction about any passenger aboard the RMS Titanic whom we knew nothing about, then the best prediction we could make would be that they did not survive. This is because we can assume that a majority of the passengers (more than 50%) did not survive the ship sinking.  
# The `predictions_0` function below will always predict that a passenger did not survive.

# In[4]:


def predictions_0(data):
    """ Model with no features. Always predicts a passenger did not survive. """

    predictions = []
    #print(data)
    for _, passenger in data.iterrows():
        
        # Predict the survival of 'passenger'
        predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_0(data)


# ### Question 1
# *Using the RMS Titanic data, how accurate would a prediction be that none of the passengers survived?*  
# **Hint:** Run the code cell below to see the accuracy of this prediction.

# In[33]:


print (accuracy_score(outcomes, predictions))


# Predictions have an accuracy of 61.62%.

# ***
# Let's take a look at whether the feature **Sex** has any indication of survival rates among passengers using the `survival_stats` function. This function is defined in the `visuals.py` Python script included with this project. The first two parameters passed to the function are the RMS Titanic data and passenger survival outcomes, respectively. The third parameter indicates which feature we want to plot survival statistics across.  
# Run the code cell below to plot the survival outcomes of passengers based on their sex.

# In[16]:


vs.survival_stats(data, outcomes, 'Sex')


# Examining the survival statistics, a large majority of males did not survive the ship sinking. However, a majority of females *did* survive the ship sinking. Let's build on our previous prediction: If a passenger was female, then we will predict that they survived. Otherwise, we will predict the passenger did not survive.  
# Fill in the missing code below so that the function will make this prediction.  
# **Hint:** You can access the values of each feature for a passenger like a dictionary. For example, `passenger['Sex']` is the sex of the passenger.

# In[5]:


def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        if passenger['Sex'] == 'female':
            predictions.append(1)
        else:
            predictions.append(0)
    
    # Return our predictions
    #print(predictions)
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)


# ### Question 2
# *How accurate would a prediction be that all female passengers survived and the remaining passengers did not survive?*  
# **Hint:** Run the code cell below to see the accuracy of this prediction.

# In[431]:


print (accuracy_score(outcomes, predictions))


# **Answer**: *Replace this text with the prediction accuracy you found above.*

# Predictions have an accuracy of 78.68%.

# In[417]:


vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])


# Examining the survival statistics, the majority of males younger than 10 survived the ship sinking, whereas most males age 10 or older *did not survive* the ship sinking. Let's continue to build on our previous prediction: If a passenger was female, then we will predict they survive. If a passenger was male and younger than 10, then we will also predict they survive. Otherwise, we will predict they do not survive.  
# Fill in the missing code below so that the function will make this prediction.  
# **Hint:** You can start your implementation of this function using the prediction code you wrote earlier from `predictions_1`.

# In[6]:


def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    #print(sum(data['Sex'] == 'female' and data['Age'] <10 ))
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        if passenger['Sex'] == 'female' or passenger['Sex'] == 'male' and passenger['Age'] < 10:
            predictions.append(1)
        else:
            predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_2(data)


# ### Question 3
# *How accurate would a prediction be that all female passengers and all male passengers younger than 10 survived?*  
# **Hint:** Run the code cell below to see the accuracy of this prediction.

# In[433]:


print (accuracy_score(outcomes, predictions))


# Predictions have an accuracy of 79.35%.

# ***
# Adding the feature **Age** as a condition in conjunction with **Sex** improves the accuracy by a small margin more than with simply using the feature **Sex** alone. Now it's your turn: Find a series of features and conditions to split the data on to obtain an outcome prediction accuracy of at least 80%. This may require multiple features and multiple levels of conditional statements to succeed. You can use the same feature multiple times with different conditions.   
# **Pclass**, **Sex**, **Age**, **SibSp**, and **Parch** are some suggested features to try.
# 
# Use the `survival_stats` function below to to examine various survival statistics.  
# **Hint:** To use mulitple filter conditions, put each condition in the list passed as the last argument. Example: `["Sex == 'male'", "Age < 18"]`

# In[57]:


vs.survival_stats(data, outcomes,'Sex',["Pclass == 3","Age <= 65", "Age >= 10"])


# After exploring the survival statistics visualization, fill in the missing code below so that the function will make your prediction.  
# Make sure to keep track of the various features and conditions you tried before arriving at your final prediction model.  
# **Hint:** You can start your implementation of this function using the prediction code you wrote earlier from `predictions_2`.

# In[50]:


def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        if  passenger['Pclass'] < 3:
            if passenger['Sex'] == 'female' or passenger['Sex'] == 'male' and passenger['Age'] < 18:
                predictions.append(1)
            else:
                predictions.append(0)
        elif passenger['Pclass'] == 3 :
            if passenger['Sex'] == 'female' and passenger['Age'] > 60 or passenger['Sex'] == 'female' and passenger['Age'] < 9:
                predictions.append(1)
            #else: #passenger['Sex'] == 'male': 
            #    predictions.append(0)
            elif passenger['Sex'] == 'female' and passenger['Age'] <= 60 and passenger['Age'] >= 9 :
                predictions.append(0)
            else: 
                predictions.append(0)
                
        else:
            predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_3(data)


# ### Question 4
# *Describe the steps you took to implement the final prediction model so that it got an accuracy of at least 80%. What features did you look at? Were certain features more informative than others? Which conditions did you use to split the survival outcomes in the data? How accurate are your predictions?*  
# **Hint:** Run the code cell below to see the accuracy of your predictions.

# In[51]:


print (accuracy_score(outcomes, predictions))


# Predictions have an accuracy of 80.58%.
# I need to look at the pclass and split the case on pclass < 3 and pclass == 3. It looks like pclass < 3 have the highest survivor rate for both females and males. In pclass = 3, I can safely assume all males don't make it and only very older and very young females make the life. Another feature is Fair.It looks like the higher it goes, the more likely the buyer survives but it is a postive correlation of pclass.

# # Conclusion
# 
# After several iterations of exploring and conditioning on the data, you have built a useful algorithm for predicting the survival of each passenger aboard the RMS Titanic. The technique applied in this project is a manual implementation of a simple machine learning model, the *decision tree*. A decision tree splits a set of data into smaller and smaller groups (called *nodes*), by one feature at a time. Each time a subset of the data is split, our predictions become more accurate if each of the resulting subgroups are more homogeneous (contain similar labels) than before. The advantage of having a computer do things for us is that it will be more exhaustive and more precise than our manual exploration above. [This link](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) provides another introduction into machine learning using a decision tree.
# 
# A decision tree is just one of many models that come from *supervised learning*. In supervised learning, we attempt to use features of the data to predict or model things with objective outcome labels. That is to say, each of our data points has a known outcome value, such as a categorical, discrete label like `'Survived'`, or a numerical, continuous value like predicting the price of a house.
# 
# ### Question 5
# *Think of a real-world scenario where supervised learning could be applied. What would be the outcome variable that you are trying to predict? Name two features about the data used in this scenario that might be helpful for making the predictions.*  

# Test automation. It will try to predict whether a new result is a regression or expected. We can predict against the tests performed before with at least two distinct configurations. 
# 
# Detection and recognition in home surveillance system. It will predict if someone that gets in the house is an intruder or belongs to the house. We can study the face features of the people in the house and also their walking gait
# 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
