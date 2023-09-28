from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    APP_HOST : str = "localhost"
    APP_PORT : int = 8000
