#!/usr/bin/env python
# coding: utf-8

# In[555]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
reg = LinearRegression(fit_intercept = False)
# import random


# In[556]:

def createX(inputfile):
    train_data = pd.DataFrame(pd.read_csv(inputfile))
    # In[557]:
    df = train_data


    # In[558]:


    remove_cols = ['batsman',
        'non_striker', 'bowler', 'dismissal_kind', 'fielder' ]
    df.drop(labels=remove_cols , axis=1 , inplace = True)


    # In[559]:
    matches = df.groupby('match_id')
    match_ids = df.match_id.unique().tolist()
    balls_list = range(48,73)
    X = []
    Y = []
    Z = []


    for i in match_ids:
        a = matches.get_group(i)
        b = a.copy()
        c = b.copy()
        score = 0
        score_by_batsmen = 0
        score_by_extras = 0
        score_after = 0
        wickets_left = 10
        b['balls_done'] = b.index + 1
        rand_ball = 48
        b.fillna(0, inplace = True)
        balls_left = 0
        #calculating X
        for i in range(b.index[0], b.index[0] + rand_ball):
            runs = b.at[i, 'total_runs']
            score += runs
            score_by_batsmen += b.at[i,'batsman_runs']
            score_by_extras += b.at[i,'extra_runs']
            if (b.at[ i , 'player_dismissed'] !=0):
                wickets_left -= 1
        current_run_rate = score/(rand_ball/6) 
        current_over = b.at[b.index[0]+rand_ball-1, 'over']
        current_ball = b.at[b.index[0]+rand_ball-1,'ball']
        overs_left = (20-(current_over))
        if (current_ball >  6):
            balls_left = 0
        elif (current_ball < 6):
            balls_left = 6 - current_ball
        overs_left += (balls_left/6)
    

        #calculating Y
        for i in range(c.index[0]+ rand_ball, c.index[-1]):
            runs = c.at[ i , 'total_runs']
            score_after += runs
        future_run_rate = score_after/(overs_left)
        Y.append(future_run_rate)
        
        #Calculating Z
        score_total = score+score_after
        list1 = [current_run_rate, overs_left,wickets_left,score_by_batsmen, score_by_extras,score, 6*overs_left+balls_left, score_total]
        X.append(list1)
        M = X
        X_Train = pd.DataFrame(X, columns = ['Current Run Rate', 'Overs left','Wickets left','Score by batsmen', 'Score by extras', 'Score', 'Balls left', 'Total score'])
        Y_Train = pd.DataFrame(Y, columns = ['Future run rate'])
        MDF = pd.DataFrame(M, columns = ['Current Run Rate', 'Overs left','Wickets left','Score by batsmen', 'Score by extras', 'Score', 'Balls left', 'Total score'])
        Z = MDF.drop(labels= ['Score', 'Balls left', 'Total score'] , axis=1 , inplace = False)
    return X_Train,Y_Train,Z  


# In[560]:
X_Train,Y_Train,Z = createX('IPL_train.csv')

#Creating Dataframes with X and Y


# Z_Train = pd.DataFrame(Z, columns = ['Score', 'Balls left', 'Total score'])
# print(Z_Train)


# In[561]:


X_train, X_test, Y_train, Y_test = train_test_split(X_Train, Y_Train, test_size=0.25, random_state=0)
# print(X_test)
# print(Y_test)
X_train1 = X_train.loc[:,['Current Run Rate', 'Overs left','Wickets left','Score by batsmen', 'Score by extras']]
X_test1 = X_test.loc[:,['Current Run Rate', 'Overs left','Wickets left','Score by batsmen', 'Score by extras']]
# print(X_train1)
# print(Y_train)
model = reg.fit(X_train1, Y_train)
# print(model.predict([X_test1.iloc[1]]))
# print(Y_test.iloc[1])


# In[562]:


#Calculating Predicted Score vs Actual Score
X_Test_Copy = X_test.copy()
Y_Test_Copy = Y_test.copy()
# Output = pd.DataFrame()
# Output['Actual Score'] = X_test.loc[:,['Total score']]
prr = np.array(model.predict(X_test1)).tolist()
# predic_score = prr*(X_test.iloc[:,'Balls Left'])
# Output['Predicted Score'] = 
# print(prr)


# In[563]:


X_test2 = X_test.copy()
X_test2.drop(labels=['Current Run Rate', 'Overs left','Wickets left','Score by batsmen', 'Score by extras'] , axis=1 , inplace = True)
#Calculating Predicted score
# print(X_test2)
X_test2['Predicted Score'] = X_test2['Score'] + (X_test['Balls left']*(prr[:][0])/6)

X_test2.drop(labels=['Score', 'Balls left'] , axis=1 , inplace = True)
# print(X_test2)


# In[564]:


from sklearn.metrics import mean_squared_error as mse
output = X_test2.copy()
# print(mse(output.loc[:,'Total score'],output.loc[:,'Predicted Score']))


# In[565]:



# plt.plot(output.loc[:,'Total score'],output.loc[:,'Predicted Score'],'r.')
# plt.plot(output.loc[:,'Total score'],output.loc[:,'Total score'])
# plt.show()


# In[ ]:
def predict_score(testfile,outputfile):
    X,Y,Z = createX(testfile)
    test_pred = model.predict(Z)
    prr = np.array(test_pred).tolist()
    X_test2 = X.copy()
    X_test2.drop(labels=['Current Run Rate', 'Overs left','Wickets left','Score by batsmen', 'Score by extras','Total score'] , axis=1 , inplace = True)
#Calculating Predicted score
# print(X_test2)
    X_test2['Predicted Score'] = X_test2['Score'] + (X['Balls left']*(prr[:][0])/6)
    X_test2.drop(labels=['Score', 'Balls left'] , axis=1 , inplace = True)  
    final_output = X_test2
    final_output.to_csv(outputfile)

predict_score('IPL_test.csv','IPL_test_predictions.csv')

# In[ ]:





# In[ ]:




