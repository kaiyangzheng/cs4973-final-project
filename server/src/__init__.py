from flask import Flask
import os
from src.config.config import Config
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_socketio import SocketIO
from flask_cors import CORS

# loading environment variables
load_dotenv()

# declaring flask application
app = Flask(__name__)

# Configure Flask to not redirect for slashes
app.url_map.strict_slashes = False

# Enable CORS with the simplest configuration
CORS(app)

# calling the dev configuration
config = Config().dev_config

# making our application to use dev env
app.env = config.ENV

# Path for our local sql lite database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("SQLALCHEMY_DATABASE_URI_DEV")

# To specify to track modifications of objects and emit signals
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = os.environ.get("SQLALCHEMY_TRACK_MODIFICATIONS")

# set secret key for the application
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")

# sql alchemy instance
db = SQLAlchemy(app)

# Flask Migrate instance to handle migrations
migrate = Migrate(app, db)

# socket io instance
socketio = SocketIO(app, cors_allowed_origins="*")

# import models to let the migrate tool know
from src.models.query_model import UserQuery

# import api blueprint to register it with app
from src.routes import api
app.register_blueprint(api, url_prefix = "/api")