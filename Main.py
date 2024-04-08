from flask import Flask,render_template,url_for,request
from flask_material import Material
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('heart.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


model = RandomForestClassifier(random_state=1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

app = Flask(__name__)
Material(app)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/page')
def page():
    return render_template("page.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/login1')
def login1():
    return render_template("login1.html")


@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/single')
def single():
    return render_template("single.html")

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset)
        return render_template("preview.html",df_view = df)


@app.route('/analyze_form')
def analyze_form():
    return render_template("analyze_form.html")

@app.route('/analyze',methods=["POST"])
def analyze():
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps'] 
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        
        model_choice = request.form['model_choice']

        sample_data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        clean_data = [float(i) for i in sample_data]

        result_prediction = model.predict(sc.transform([clean_data]))
        


    return render_template('result.html',
                           age = age, sex = sex,cp = cp,trestbps = trestbps,
                           chol = chol,fbs = fbs,restecg = restecg,thalach = thalach,
                           exang = exang,oldpeak = oldpeak,slope = slope,ca = ca,
                           thal = thal,sample_data1=sample_data,
                           result_prediction=result_prediction,model_selected=model_choice)


if __name__ == '__main__':
	app.run(debug=True)
