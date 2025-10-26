from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import openai
import os

app = Flask(__name__)

# Load a simple keras model for demonstration purposes
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({'prediction': float(prediction[0][0])})

@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data['prompt']

    # Call OpenAI API
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        model="gpt-4o-mini",
        message=[{"role": "user", "content": prompt}]
    )

    result = response.choices[0].message.content.strip()
    return jsonify({'response':result})

if __name__ == '__main__':
    app.run(debug=True)



