from flask import request, Response, json, Blueprint
from src.models.query_model import UserQuery

# user controller blueprint to be registered with api blueprint
queries = Blueprint("queries", __name__)

@queries.route("/", methods=["GET"])
def get_queries():
    """
    Get all queries
    """
    # get all queries from the database
    queries = UserQuery.query.all()

    # convert to json
    queries_json = [query.to_dict() for query in queries]

    # return the json response
    return Response(json.dumps(queries_json), mimetype="application/json")

@queries.route("/", methods=["POST"])
def add_query():
    """
    Add a new query
    """
    # get the data from the request
    data = request.get_json()
    
    # call llm service here with asyncio

    # create a new query object
    query = UserQuery(prompt=data["prompt"], response=data["response"])

    # add the query to the database
    query.save()

    # return the json response
    return Response(json.dumps(query.to_dict()), mimetype="application/json")