from flask import Flask, jsonify, request
import lief
import sys
import typing
import numpy as np
from learning.ensemble.Models.ClassificationResult import ClassificationResult

MAX_BYTES = 2097152


def create_app(model, predScanners, statefulDefense):
    app = Flask(__name__)
    app.config['model'] = model
    app.config['predScanners'] = predScanners
    app.config['statefulDefense'] = statefulDefense

    # analyse a sample
    @app.route('/', methods=['POST'])
    def post():
        # curl -XPOST --data-binary @somePEfile http://127.0.0.1:8080/ -H "Content-Type: application/octet-stream"
        if request.headers['Content-Type'] != 'application/octet-stream':
            resp = jsonify({'error': 'expecting application/octet-stream'})
            resp.status_code = 400  # Bad Request
            return resp

        bytez = request.data
        if len(bytez) == 0:
            # empty files are considered benign (can't contain malicious code)
            return create_response(0)
        if len(bytez) > MAX_BYTES:
            # too large files are treated as malicious
            return create_response(1)

        pe_binary = None
        try:
            pe_binary = lief.parse(bytez)
        except Exception:
            pass

        # PE files raising parse errors are treated as malicious
        # since lief sometimes doesn't raise an error but returns a None obj, we catch that as well
        if pe_binary is None:
            return create_response(1)

        model = app.config['model']
        predScanners = app.config['predScanners']
        statefulDefense = app.config['statefulDefense']

        # A. Pre-scan
        prescan_results: list = iterate_over_predators(predscanners=predScanners, bytez=bytez, pe_binary=pe_binary)
        if sum(prescan_results) > 0:
            return create_response(1)

        # B. Classify
        # result = model.predict(bytez, pe_binary=pe_binary)
        # aggregate predictions
        # result = int(max(result) > 0)
        response_classification: typing.List[ClassificationResult] = model.predict_all_proba(bytez, pe_binary=pe_binary)
        result = ClassificationResult.get_max_decision(response_classification)

        # C. Stateful defense
        if len(response_classification) > 0 and response_classification[0].features is not None:
            stateful_response: tuple = statefulDefense.check(
                score=np.max([x.prob_score for x in response_classification]),
                ismalware=result,
                features=response_classification[0].features,
                bytez=bytez, pe_binary=pe_binary)

            if stateful_response[0] is True:
                print(stateful_response[0], stateful_response[1], file=sys.stderr)
                return create_response(result=1)
        else:
            print("Error in activating stateful defense: {}".format(len(response_classification)), file=sys.stderr)
        # print(result, [x.prob_score for x in response_classification], np.max([x.prob_score for x in response_classification]),
        #       file=sys.stderr)

        if not isinstance(result, int) or result not in {0, 1}:
            resp = jsonify({'error': 'unexpected model result (not in [0,1])'})
            resp.status_code = 500  # Internal Server Error
            return resp

        return create_response(result=result)

    return app


def create_response(result, status_code=200):
    resp = jsonify({'result': result})
    resp.status_code = status_code
    return resp


def iterate_over_predators(predscanners: list, bytez, pe_binary):
    preds = []
    for predScanner in predscanners:
        try:
            respPre = predScanner.check_file(bytez=bytez, lief_binary=pe_binary)
        except Exception as e:
            print('Predator Error:', str(e), file=sys.stderr)
            respPre = False
        preds.append(respPre)
    return preds
