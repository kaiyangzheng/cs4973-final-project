from flask import request, Response, json, Blueprint

# user controller blueprint to be registered with api blueprint
users = Blueprint("users", __name__)

# route for login api/users/signin
@users.route('/signin', methods = ["GET", "POST"])
def handle_login():
    return Response(
        response=json.dumps({'status': "success"}),
        status=200,
        mimetype='application/json'
    )

# route for login api/users/signup
@users.route('/signup', methods = ["GET", "POST"])
def handle_signup():
    return Response(
        response=json.dumps({'status': "success"}),
        status=200,
        mimetype='application/json'
    )