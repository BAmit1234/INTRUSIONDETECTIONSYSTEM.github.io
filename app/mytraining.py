import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report#,confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd
    
# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    

if __name__=="__main__":   
    
    df = pd.read_csv("kddcup.data_10_percent_corrected", header=None)

    df.columns = [
        'duration',
        'protocol_type',
        'service',
        'flag',
        'src_bytes',
        'dst_bytes',
        'land',
        'wrong_fragment',
        'urgent',
        'hot',
        'num_failed_logins',
        'logged_in',
        'num_compromised',
        'root_shell',
        'su_attempted',
        'num_root',
        'num_file_creations',
        'num_shells',
        'num_access_files',
        'num_outbound_cmds',
        'is_host_login',
        'is_guest_login',
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'outcome'
    ]

    
# Now encode the feature vector
    df.dropna(inplace=True,axis=1)
    encode_numeric_zscore(df, 'duration')
    encode_text_dummy(df, 'protocol_type')
    encode_text_dummy(df, 'service')
    encode_text_dummy(df, 'flag')
    encode_numeric_zscore(df, 'src_bytes')
    encode_numeric_zscore(df, 'dst_bytes')
    encode_text_dummy(df, 'land')
    encode_numeric_zscore(df, 'wrong_fragment')
    encode_numeric_zscore(df, 'urgent')
    encode_numeric_zscore(df, 'hot')
    encode_numeric_zscore(df, 'num_failed_logins')
    encode_text_dummy(df, 'logged_in')
    encode_numeric_zscore(df, 'num_compromised')
    encode_numeric_zscore(df, 'root_shell')
    encode_numeric_zscore(df, 'su_attempted')
    encode_numeric_zscore(df, 'num_root')
    encode_numeric_zscore(df, 'num_file_creations')
    encode_numeric_zscore(df, 'num_shells')
    encode_numeric_zscore(df, 'num_access_files')
    encode_numeric_zscore(df, 'num_outbound_cmds')
    encode_text_dummy(df, 'is_host_login')
    encode_text_dummy(df, 'is_guest_login')
    encode_numeric_zscore(df, 'count')
    encode_numeric_zscore(df, 'srv_count')
    encode_numeric_zscore(df, 'serror_rate')
    encode_numeric_zscore(df, 'srv_serror_rate')
    encode_numeric_zscore(df, 'rerror_rate')
    encode_numeric_zscore(df, 'srv_rerror_rate')
    encode_numeric_zscore(df, 'same_srv_rate')
    encode_numeric_zscore(df, 'diff_srv_rate')
    encode_numeric_zscore(df, 'srv_diff_host_rate')
    encode_numeric_zscore(df, 'dst_host_count')
    encode_numeric_zscore(df, 'dst_host_srv_count')
    encode_numeric_zscore(df, 'dst_host_same_srv_rate')
    encode_numeric_zscore(df, 'dst_host_diff_srv_rate')
    encode_numeric_zscore(df, 'dst_host_same_src_port_rate')
    encode_numeric_zscore(df, 'dst_host_srv_diff_host_rate')
    encode_numeric_zscore(df, 'dst_host_serror_rate')
    encode_numeric_zscore(df, 'dst_host_srv_serror_rate')
    encode_numeric_zscore(df, 'dst_host_rerror_rate')
    encode_numeric_zscore(df, 'dst_host_srv_rerror_rate') 
    df.dropna(inplace=True,axis=1)
    
    df_temp = df.copy()
    y = df_temp['outcome']
    df_temp = df_temp.drop('outcome', 1)
    X_train, X_test, y_train, y_test = train_test_split(df_temp, y, test_size=0.2, random_state=42)
     # open a file, where you want to store data
    file=open('data.pkl','wb')
    
    #dump information into the file
    pickle.dump([X_train, X_test, y_train, y_test], file)
    file.close()
    df.dropna(inplace=True,axis=1)
#print('---------------------Classifying Using Decesion Tree Model-----------------')

    gini=DecisionTreeClassifier(criterion='gini',random_state=100, max_depth=5,min_samples_leaf=5)
    gini.fit(X_train,y_train)
    # open a file, where you want to store data
    file=open('gini.pkl','wb')
    
    #dump information into the file
    pickle.dump(gini, file)
    file.close()
    pred=gini.predict(X_test)

    
    
#print("-------------Decesion tree using Gini Index-------------------")
  
    print("gini accuracy score",accuracy_score(y_test,pred))
#print("confusion matrix:",confusion_matrix(y_test,pred))     'unncomment it if you want to see the confusion matrix'
    print('classification report ',classification_report(y_test,pred))






#("-------Decesion Tree using Entropy--------------------------")
    entropy=DecisionTreeClassifier(criterion='entropy',random_state=100, max_depth=5,min_samples_leaf=5)

    entropy.fit(X_train,y_train)
    
    # open a file, where you want to store data
    file=open('entropy.pkl','wb')
    
    #dump information into the file
    pickle.dump(entropy, file)
    
    file.close()
    pred1=entropy.predict(X_test)
    print("Entropy accuracy score",accuracy_score(y_test,pred1))


#print("confusion matrix:",confusion_matrix(y_test,pred1))     'unncomment it if you want to see the confusion matrix'
    print('classification report ',classification_report(y_test,pred1))




#('--------------------Classifying Using Random Forest-----------------')

    RandomForest = RandomForestClassifier(n_estimators=50)
    RandomForest.fit(X_train,y_train)
    
    # open a file, where you want to store data
    file=open('RandomForest.pkl','wb')
    
    #dump information into the file
    pickle.dump(RandomForest, file)
    
    file.close()
    pred2=RandomForest.predict(X_test)
    print("Random Forest accuracy score",accuracy_score(y_test,pred2))
    print('classification report ',classification_report(y_test,pred2))

    file=open('pre.pkl','wb')
    
    #dump information into the file
    pickle.dump([pred,pred1,pred2], file)
    
    file.close()
