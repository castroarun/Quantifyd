"""Register all SCC route blueprints."""


def register_blueprints(app):
    from .dashboard import bp as dashboard_bp
    from .positions import bp as positions_bp
    from .blueprints import bp as blueprints_bp
    from .deep_dive import bp as deep_dive_bp
    from .strategies import bp as strategies_bp
    from .playbook import bp as playbook_bp
    from .scc_api import bp as api_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(positions_bp)
    app.register_blueprint(blueprints_bp)
    app.register_blueprint(deep_dive_bp)
    app.register_blueprint(strategies_bp)
    app.register_blueprint(playbook_bp)
    app.register_blueprint(api_bp)
