from flask import Blueprint, request, jsonify

from app.api.v1.services import (
    process_audio_request,
    predict_genres,
    predict_instruments,
    predict_vocals,
)

v1 = Blueprint("v1", __name__, url_prefix="/api/v1")


@v1.route("/predict/genres", methods=["POST"])
def genres():
    audio_data = process_audio_request(request)
    top_n = request.args.get("top_n", default=5, type=int)
    results = predict_genres(audio_data, top_n=top_n)
    return jsonify({"predictions": results})


@v1.route("/predict/instruments", methods=["POST"])
def instruments():
    audio_data = process_audio_request(request)
    top_n = request.args.get("top_n", default=5, type=int)
    results = predict_instruments(audio_data, top_n=top_n)
    return jsonify({"predictions": results})


@v1.route("/predict/vocals", methods=["POST"])
def vocals():
    audio_data = process_audio_request(request)
    results = predict_vocals(audio_data)
    return jsonify({"prediction": results})
