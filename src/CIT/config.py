import os
from dynaconf import Dynaconf

config_dir = os.path.dirname(os.path.abspath(__file__))

settings_file_path = os.path.join(config_dir, "settings.toml")
secret_file_path = os.path.join(config_dir, "secrets.toml")

settings = Dynaconf(
    envvar_prefix="CIT",
    settings_files=[settings_file_path, secret_file_path],
)
