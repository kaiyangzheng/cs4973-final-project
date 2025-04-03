from flask import request, Response, json, Blueprint, current_app
from src.models.query_model import UserQuery
import asyncio
import threading
from src.services.llm_service import process_user_query
from src import app, db

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

def run_async_task(flask_app, query_id, prompt, paper_content):
    """Run an asyncio coroutine in a separate thread"""
    print(f"\n==== Starting async task for query_id: {query_id} ====")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Has paper content: {paper_content is not None}")
    
    # Create a Flask application context for this thread
    with flask_app.app_context():
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                print(f"Creating event loop for query {query_id}")
                # Run the coroutine in the new loop
                print(f"Calling process_user_query for query {query_id}")
                result = loop.run_until_complete(process_user_query(prompt, paper_content))
                print(f"Query {query_id} - Got result: {result.keys() if result else 'None'}")
                
                # Get the query from the database
                query = UserQuery.query.get(query_id)
                if query:
                    # Update the query with the response and paper categories
                    print(f"Updating query {query_id} with response of length: {len(result.get('response', ''))}")
                    query.response = result.get("response", "No response received")
                    query.pending = False
                    
                    # Store paper categories as JSON string if they exist
                    if 'paper_categories' in result and result['paper_categories']:
                        query.paper_categories = json.dumps(result['paper_categories'])
                        print(f"Added paper categories for query {query_id}: {result['paper_categories']}")
                    
                    db.session.commit()
                    print(f"Successfully updated query {query_id}, pending={query.pending}")
                else:
                    print(f"ERROR: Query {query_id} not found in database")
            finally:
                # Clean up
                loop.close()
                print(f"Closed event loop for query {query_id}")
        except Exception as e:
            print(f"Thread error for query {query_id}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(traceback.format_exc())
            # Try to update the query with the error
            with flask_app.app_context():
                try:
                    query = UserQuery.query.get(query_id)
                    if query:
                        query.response = f"Error processing request: {str(e)}"
                        query.pending = False
                        db.session.commit()
                        print(f"Updated query {query_id} with error message")
                    else:
                        print(f"ERROR: Query {query_id} not found when trying to update with error")
                except Exception as inner_e:
                    print(f"Error updating query with error for query {query_id}: {str(inner_e)}")
                    
    print(f"==== Completed async task for query_id: {query_id} ====\n")

@queries.route("/", methods=["POST"])
def add_query():
    """
    Add a new query
    """
    # get the data from the request
    data = request.get_json()
    
    # Create a new query object with pending status
    query = UserQuery(prompt=data["prompt"], response="", pending=True)
    
    # Save the query to get an ID
    db.session.add(query)
    db.session.commit()
    
    query_id = query.id
    prompt = data["prompt"]
    paper_content = data.get("paper_content")
    
    # Print before thread creation
    print(f"About to create thread for query {query_id}")
    
    try:
        # Start a background thread to process the query
        thread = threading.Thread(
            target=run_async_task, 
            args=(app, query_id, prompt, paper_content)
        )
        thread.daemon = True
        thread.start()
        print(f"Thread for query {query_id} started successfully")
    except Exception as e:
        print(f"ERROR creating thread: {str(e)}")
        # Update the query with the error
        query.response = f"Error starting processing thread: {str(e)}"
        query.pending = False
        db.session.commit()
    
    # Return the query immediately with pending status
    return Response(json.dumps(query.to_dict()), mimetype="application/json")

@queries.route("/", methods=["DELETE"])
def clear_queries():
    """
    Clear all queries from the database
    """
    try:
        # Delete all queries
        UserQuery.query.delete()
        db.session.commit()
        return Response(json.dumps({"success": True, "message": "All queries cleared successfully"}), mimetype="application/json")
    except Exception as e:
        db.session.rollback()
        return Response(json.dumps({"success": False, "message": f"Error clearing queries: {str(e)}"}), mimetype="application/json", status=500)