from flask import Flask, jsonify, request
from flask_cors import CORS

from codes.ml import SantosMLEvaluator
from santos import Santos

app = Flask(__name__)
CORS(app)

santos_model = Santos()
ml_model = SantosMLEvaluator()


@app.route('/sending', methods=['POST'])
def sending():
    file_name = request.get_json()['fileName']
    print("开始查询", file_name, "相关的数据...")
    santos_model.start_query(file_name)
    return jsonify({"status": "success"})


@app.route('/listing', methods=['GET'])
def listing():
    result = santos_model.get_result()
    return jsonify(result[0])


@app.route('/training', methods=['POST'])
def training():
    global ml_model
    file_name = santos_model.file_name
    target = request.get_json()['postList'][1]
    model_name = request.get_json()['postList'][2]
    task = request.get_json()['postList'][3]

    ml_model.set_query_table(file_name)
    ml_model.set_similar_tables(santos_model.get_result()[0])

    ml_model.run(target_column=target, task_type=task, model_type=model_name)
    return jsonify({"status": "success"})


@app.route('/progress', methods=['GET'])
def progress():
    global ml_model
    result = ml_model.get_progress()
    return jsonify(result)


if __name__ == "__main__":
    app.run(port=3001)
