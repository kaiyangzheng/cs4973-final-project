from src import db

class UserQuery(db.Model):
    id = db.Column(db.Integer(), primary_key = True, unique=True)
    prompt = db.Column(db.String(), nullable=False)
    response = db.Column(db.String(), nullable=False, default="")
    pending = db.Column(db.Boolean(), default=True)
    created_at = db.Column(db.DateTime(), server_default=db.func.now())
    updated_at = db.Column(db.DateTime(), server_default=db.func.now(), onupdate=db.func.now())