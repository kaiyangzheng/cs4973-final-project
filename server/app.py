from src import config, app

if __name__ == "__main__":
    app.run(host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG)
    
# source: https://ashleyalexjacob.medium.com/flask-api-folder-guide-2023-6fd56fe38c00