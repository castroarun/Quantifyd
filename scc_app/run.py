"""Entry point for the SCC Flask app."""
from scc_app import create_app
from scc_app.config import SCC_PORT, SCC_DEBUG

app = create_app()

if __name__ == '__main__':
    print(f"\n  Strategy Command Center running at http://127.0.0.1:{SCC_PORT}\n")
    app.run(host='127.0.0.1', port=SCC_PORT, debug=SCC_DEBUG)
