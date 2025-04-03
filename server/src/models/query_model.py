from src import db
from datetime import datetime
import json

class UserQuery(db.Model):
    id = db.Column(db.Integer(), primary_key = True, unique=True)
    prompt = db.Column(db.String(), nullable=False)
    response = db.Column(db.String(), nullable=False, default="")
    pending = db.Column(db.Boolean(), default=True)
    paper_categories = db.Column(db.String(), nullable=True)  # Store as JSON string
    created_at = db.Column(db.DateTime(), server_default=db.func.now())
    updated_at = db.Column(db.DateTime(), server_default=db.func.now(), onupdate=db.func.now())
    
    def to_dict(self):
        """
        Convert the model to a dictionary for JSON serialization
        """
        return {
            'id': self.id,
            'prompt': self.prompt,
            'response': self.response,
            'pending': self.pending,
            'paper_categories': json.loads(self.paper_categories) if self.paper_categories else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def save(self):
        """
        Save the model to the database
        """
        if not self.id:
            db.session.add(self)
        
        db.session.commit()
        
        return self