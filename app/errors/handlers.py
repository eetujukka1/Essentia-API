from flask import jsonify


def register_error_handlers(app):
    @app.errorhandler(ValueError)
    def handle_value_error(e):
        return jsonify({"error": str(e)}), 400

    @app.errorhandler(400)
    def handle_bad_request(e):
        return jsonify({"error": "Bad request"}), 400

    @app.errorhandler(404)
    def handle_not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def handle_internal_error(e):
        return jsonify({"error": "Internal server error"}), 500
