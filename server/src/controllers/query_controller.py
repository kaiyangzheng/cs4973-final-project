from flask import request, Response, json, Blueprint, current_app
from src.models.query_model import UserQuery
import asyncio
import threading
from src.services.llm_service import process_user_query
from src import app, db, socketio
from flask_socketio import emit

# user controller blueprint to be registered with api blueprint
queries = Blueprint("queries", __name__)

# Dictionary to store active socket connections
active_sockets = {}

@socketio.on("connect")
def handle_connect():
    print("Client connected")
    client_id = request.sid
    active_sockets[client_id] = True
    print(f"Client ID: {client_id}")
    
@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")
    client_id = request.sid
    print(f"Client ID: {client_id}")
    
    # Remove from active sockets dictionary
    if client_id in active_sockets:
        del active_sockets[client_id]
    
    # Delete all queries associated with this socket ID
    try:
        queries_to_delete = UserQuery.query.filter_by(socket_id=client_id).all()
        for query in queries_to_delete:
            db.session.delete(query)
        db.session.commit()
        print(f"Deleted all queries for disconnected client {client_id}")
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting queries for client {client_id}: {str(e)}")

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

def run_async_task(flask_app, query_id, prompt, paper_content, socket_id):
    """Run an asyncio coroutine in a separate thread"""
    print(f"\n==== Starting async task for query_id: {query_id}, socket_id: {socket_id} ====")
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
                # Pass the socket_id to process_user_query
                result = loop.run_until_complete(process_user_query(prompt, paper_content, socket_id))
                print(f"Query {query_id} - Got result: {result.keys() if result else 'None'}")

                # Get the query from the database
                query = UserQuery.query.get(query_id)
                if query:
                    # Update the query with the response and paper categories
                    print(f"Updating query {query_id} with response of length: {len(result.get('response', ''))}")
                    query.response = result.get("response", "No response received")
                    query.pending = False

                    # Check for categories in either key (preferring 'categories' if available)
                    categories = None
                    if 'categories' in result and result['categories']:
                        categories = result['categories']
                        print(f"Found categories in 'categories' key: {categories}")
                    elif 'paper_categories' in result and result['paper_categories']:
                        categories = result['paper_categories']
                        print(f"Found categories in 'paper_categories' key: {categories}")
                    
                    # Store paper categories as JSON string if they exist
                    if categories:
                        query.paper_categories = json.dumps(categories)
                        print(f"Added categories for query {query_id}: {categories}")

                    db.session.commit()
                    print(f"Successfully updated query {query_id}, pending={query.pending}")
                    
                    # Emit the result to the specific socket ID
                    if socket_id in active_sockets:
                        response_data = query.to_dict()
                        print(f"Emitting response with categories: {response_data.get('categories')}")
                        socketio.emit('query_response', response_data, room=socket_id)
                        print(f"Emitted response to socket {socket_id}")
                    else:
                        print(f"Socket {socket_id} is no longer active, not emitting response")
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
                        
                        # Emit error to the specific socket ID
                        if socket_id in active_sockets:
                            socketio.emit('query_response', query.to_dict(), room=socket_id)
                            print(f"Emitted error response to socket {socket_id}")
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
    
    # Get the socket ID from the request if available, or use a default
    socket_id = data.get("socket_id")
    
    # Create a new query object with pending status and socket_id
    query = UserQuery(
        prompt=data["prompt"], 
        response="", 
        pending=True,
        socket_id=socket_id
    )

    # Save the query to get an ID
    db.session.add(query)
    db.session.commit()

    query_id = query.id
    prompt = data["prompt"]
    paper_content = data.get("paper_content")

    # Print before thread creation
    print(f"About to create thread for query {query_id} with socket_id {socket_id}")

    try:
        # Start a background thread to process the query
        thread = threading.Thread(
            target=run_async_task, 
            args=(app, query_id, prompt, paper_content, socket_id)
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
        
        # Emit error to the specific socket ID
        if socket_id in active_sockets:
            socketio.emit('query_response', query.to_dict(), room=socket_id)

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