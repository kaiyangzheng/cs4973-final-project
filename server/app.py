from src import config, app, socketio

if __name__ == "__main__":
    socketio.run(app,
            host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG)
    
# source: https://ashleyalexjacob.medium.com/flask-api-folder-guide-2023-6fd56fe38c00