
from flask import Flask,render_template,request,jsonify
from flask_cors import CORS
from chat2 import chat

app = Flask(__name__)
CORS(app)

@app.route("/",methods = ["GET"])
def index_get():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    text = request.get_json().get("message")
    response = chat(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=True)


