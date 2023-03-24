import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import turicreate as tc

os.chdir('/Users/tomleung/Downloads/ITE3905_AI_project/webApplication')
app = Flask(__name__)
CORS(app)

def image_Detection(filename):
    model = tc.load_model('../HongKong-dollar.model')
    image_data = tc.image_analysis.load_images(os.getcwd() + '/' + filename)
    prediction = model.predict(image_data, output_type='class')
    return prediction[0]

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = file.filename
    file.save(os.getcwd() + '/' + filename)
    prediction = image_Detection(filename)
    os.remove(os.getcwd() + '/' + filename)
    return jsonify({"message": prediction})

if __name__ == '__main__':
    app.run(debug=True, port=8000)