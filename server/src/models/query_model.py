from src import db
from datetime import datetime

class Query(db.Model):
    id = db.Column(db.Integer(), primary_key=True, unique=True, nullable=False)
    query = db.Column(db.String(), nullable=False)
    response = db.Column(db.String(), default="", nullable=False)
    pending = db.Column(db.Boolean(), default=True, nullable=False)
    created_at = db.Column(db.DateTime(), default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime(), default=datetime.now, onupdate=datetime.now, nullable=False) 