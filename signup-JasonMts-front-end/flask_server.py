from flask import Flask
from flask import render_template, abort, jsonify, request,redirect, json
from example_ml import analyzer
app = Flask(__name__)
app.debug = True

@app.route('/')
@app.route("/index")
def index():
    return render_template('index.html')

@app.route('/learning', methods=['POST'])
def learning():
    #loads requested data sent form frontend
    data = json.loads(request.data)
    response = analyzer(data)
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)