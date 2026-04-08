"""Strategy Command Center — Flask App Factory."""
from flask import Flask
from .config import PARENT_DIR
import sys

# Ensure parent services are importable
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))


def create_app():
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')
    app.secret_key = 'scc-quantifyd-2026'

    # Register blueprints
    from .routes import register_blueprints
    register_blueprints(app)

    return app
