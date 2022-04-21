from flask import Flask, render_template, url_for, request
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/quiz')
def quiz():
    return render_template('Quiz.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():

    int_features = [eval(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    input_data_reshaped = final_features


    prediction = model.predict(input_data_reshaped)

    return render_template('after.html', data= prediction)

if __name__ == '__main__':
    app.run(debug=True)