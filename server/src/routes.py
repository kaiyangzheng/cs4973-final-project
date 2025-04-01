from flask import Blueprint
from src.controllers.query_controller import queries

# main blueprint to be registered with application
api = Blueprint('api', __name__)

# register user with api blueprint
api.register_blueprint(queries, url_prefix="/queries")