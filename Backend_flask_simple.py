from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return "Hello, World! The app is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
