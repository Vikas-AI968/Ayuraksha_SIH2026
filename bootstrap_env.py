"""Load environment variables from .env before any other module reads os.environ.

Import this as the FIRST import in every process entry point (api/main.py,
every scripts/*.py, tests/conftest.py) -- before importing anything else from
this project -- since several modules read configuration from os.environ at
import time (e.g. embeddings/provider.py's FASTEMBED_INIT_TIMEOUT_SECONDS).
If .env is loaded after those imports, they never see the values.

python-dotenv's load_dotenv() does not override variables already set in the
real process environment, so real environment variables (e.g. exported in a
shell, or injected by a deployment platform) always take precedence over
.env file contents.
"""
from dotenv import load_dotenv

load_dotenv()
