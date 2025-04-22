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

# Load embeddings into vector database
from src.services.vector_db import vector_db
base_dir = os.path.dirname(os.path.dirname(__file__))

# Check for semantic embeddings first (preferred)
semantic_embeddings_file = os.path.join(base_dir, 'cs_papers_semantic_embeddings.pkl')
standard_embeddings_file = os.path.join(base_dir, 'cs_papers_embeddings.pkl')

# Try loading semantic embeddings first, then fall back to standard embeddings
if os.path.exists(semantic_embeddings_file):
    print(f"Loading semantic embeddings from {semantic_embeddings_file}")
    vector_db.load_embeddings(semantic_embeddings_file)
    print(f"Loaded {len(vector_db.vectors)} semantic embeddings with {len(vector_db.categories)} categories")
elif os.path.exists(standard_embeddings_file):
    print(f"Semantic embeddings not found. Loading standard embeddings from {standard_embeddings_file}")
    vector_db.load_embeddings(standard_embeddings_file)
    print(f"Loaded {len(vector_db.vectors)} standard embeddings with {len(vector_db.categories)} categories")
else:
    print(f"Warning: No embeddings file found. Vector database is empty.")

# import models to let the migrate tool know
from src.models.query_model import UserQuery

# import api blueprint to register it with app
from src.routes import api
app.register_blueprint(api, url_prefix = "/api")