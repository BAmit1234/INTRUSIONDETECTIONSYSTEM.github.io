from flask import Flask, render_template
from sklearn.metrics import accuracy_score,classification_report#,confusion_matrix 
app=Flask(__name__)
import pickle

file=open("entropy.pkl",'rb')
entropy=pickle.load(file)
file.close()
file=open("RandomForest.pkl",'rb')
RandomForest=pickle.load(file)
file.close()
file=open("gini.pkl",'rb')
gini=pickle.load(file)
file.close()
file=open("data.pkl",'rb')
X_train, X_test, y_train, y_test=pickle.load(file)
file.close()
file=open("pre.pkl",'rb')
pred, pred1, pred2=pickle.load(file)
file.close()
@app.route("/")
def home():
    
   return render_template("index.html")
@app.route("/random.html")
def random():
    accuracy=accuracy_score(y_test,pred2)
    return render_template("random.html",rnf=accuracy)
@app.route("/entropy.html")
def entropy():
    accuracy=accuracy_score(y_test,pred1)
    return render_template("entropy.html",enf=accuracy)
@app.route("/gini.html")
def gini():
    
    accuracy=accuracy_score(y_test,pred)
    return render_template("gini.html",gnf=accuracy)
if __name__=="__main__":
     app.run(debug=True)
