from flask import Flask,render_template,redirect,request,url_for
import numpy as np
import pickle


filename='diabetes-prediction-rfc-model.pkl'
classifier=pickle.load(open(filename,'rb'))
                       
app=Flask(__name__)

@app.route('/',methods=["POST","GET"])
def home():
    if request.method=='POST':
        preg=int(request.form['Pregnancies'])
        glucose=int(request.form['Glucose Level'])
        bp=int(request.form['Blood Pressure'])
        st=int(request.form['Skin Thickness'])
        ins=float(request.form['Insulin'])
        bmi=float(request.form['BMI'])
        dpf=float(request.form['Diabetes PF'])
        age=int(request.form['Age'])
        
        data=np.array([[preg,glucose,bp,st,ins,bmi,dpf,age]])
        my_prediction=classifier.predict(data)

        return render_template('result.html',prediction=my_prediction)

    
    
    return render_template("index.html")


    


if __name__=='__main__':
    app.run(debug=True)