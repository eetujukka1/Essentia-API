from flask import Flask

from app.config import Config
from app.api.v1.routes import v1
from app.errors.handlers import register_error_handlers


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    app.register_blueprint(v1)
    register_error_handlers(app)

    return app
