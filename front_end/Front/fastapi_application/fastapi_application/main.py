from configuration.config import AppConfig
from source.apis.api import app
from uvicorn import Server, Config


if __name__ == '__main__':
    config = AppConfig()
    server = Server(
        Config(app=app, host=config.APP_HOST, port=config.APP_PORT))
    server.run()

