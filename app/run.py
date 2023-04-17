from flask import Flask
import os, sys
import logging
FORMAT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
logging.basicConfig(format=FORMAT)



def create_app(config_name=None):

    # Initialize flask app
    app = Flask(__name__)
    logging.warning(f"--> Creating Server App with config: '{config_name}'")

    if app.config['DEBUG']:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    FORMAT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
    logging.basicConfig(format=FORMAT)

    # Registering all the blueprints
    from site_routes import site
    app.register_blueprint(site)

    return app


# Launch the web server
if __name__ == '__main__': # pragma: no cover
    app = create_app()
    debug = True
    app.run(debug=debug, host="0.0.0.0", port=int(os.environ.get("PORT", 8087)))
