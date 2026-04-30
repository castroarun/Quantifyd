"""
Covered Calls Backtester - Flask Application
=============================================

Web application for backtesting covered call strategies on Indian stocks.

Routes:
- / : Landing page with login status
- /login : Redirect to Zerodha OAuth
- /zerodha/callback : OAuth callback handler
- /logout : Clear session
- /backtest : Main backtest configuration page
- /data-management : Data download and management
- /api/* : API endpoints for backtest execution and status
"""

import os
import json
import logging
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash, Response
)
from flask_session import Session
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

# Load environment variables
load_dotenv()

# Import services
from services import (
    get_kite, get_access_token, save_access_token, get_login_url,
    is_authenticated, get_data_manager, get_backtest_db,
    CoveredCallEngine, BacktestConfig, StrikeMethod, ExitStrategy,
    NIFTY_50, TOP_10_LIQUID, FNO_LOT_SIZES,
    get_holdings, get_fundamentals, get_historical_prices,
    get_portfolio_summary, format_currency,
    portfolio_chat, clear_chat_history, get_suggested_questions,
)
from services.cpr_covered_call_service import CPRCoveredCallEngine, CPRBacktestConfig
from services.intraday_data_bridge import get_intraday_bridge
from config import (
    FLASK_SECRET_KEY, KITE_API_KEY, KITE_API_SECRET,
    STRIKE_METHODS, EXIT_RULES, RISK_FREE_RATE, DATA_DIR,
    MQ_DEFAULTS, KC6_DEFAULTS, MARUTHI_DEFAULTS, BNF_DEFAULTS, NAS_DEFAULTS, NAS_ATM_DEFAULTS,
    NAS_ATM2_DEFAULTS, NAS_ATM4_DEFAULTS,
    NAS_916_OTM_DEFAULTS, NAS_916_ATM_DEFAULTS,
    NAS_916_ATM2_DEFAULTS, NAS_916_ATM4_DEFAULTS,
    ORB_DEFAULTS,
)

# MQ Agent imports (lazy-loaded in route handlers to avoid startup overhead)
_mq_agents_loaded = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Flask App Configuration
# =============================================================================

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# Session configuration
SESSION_DIR = Path(__file__).parent / 'flask_session'
SESSION_DIR.mkdir(exist_ok=True)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = str(SESSION_DIR)
app.config['SESSION_PERMANENT'] = False
Session(app)

# Initialize Flask-SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Background scheduler for async tasks
scheduler = BackgroundScheduler()
scheduler.start()

# In-memory task status storage
task_status = {}

# Ticker service import (lazy load to avoid circular imports)
_ticker_service = None


# =============================================================================
# Decorators
# =============================================================================

def login_required(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# =============================================================================
# Authentication Routes
# =============================================================================


# =============================================================================
# Quantifyd v2 — React SPA (new design system)
# Serves the React app at /app/* — client-side routing handled by React Router
# =============================================================================
from flask import send_from_directory

@app.route('/app')
@app.route('/app/')
@app.route('/app/<path:subpath>')
def serve_react_app(subpath=''):
    """Serve React SPA. All /app/* routes return index.html for client-side routing.
    Assets under /app/assets/* are served directly."""
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'app')
    # Assets are served via /app/assets/* — handle them explicitly
    if subpath.startswith('assets/'):
        return send_from_directory(app_dir, subpath)
    # Favicon or other static files at /app/<file>
    if subpath and os.path.exists(os.path.join(app_dir, subpath)) and not os.path.isdir(os.path.join(app_dir, subpath)):
        return send_from_directory(app_dir, subpath)
    # All other paths: serve index.html (React Router handles routing)
    return send_from_directory(app_dir, 'index.html')


@app.route('/')
def index():
    """Landing page with login status"""
    authenticated = is_authenticated()
    user_name = session.get('user_name', 'User')

    return render_template(
        'index.html',
        authenticated=authenticated,
        user_name=user_name
    )


@app.route('/login')
def login():
    """Redirect to Zerodha OAuth login"""
    try:
        login_url = get_login_url()
        logger.info(f"Redirecting to Zerodha login: {login_url}")
        return redirect(login_url)
    except Exception as e:
        logger.error(f"Login error: {e}")
        flash(f'Login error: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/zerodha/callback')
def zerodha_callback():
    """Handle OAuth callback from Zerodha"""
    request_token = request.args.get('request_token')

    if not request_token:
        flash('No request token received', 'error')
        return redirect(url_for('index'))

    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=KITE_API_KEY)

        # Generate session
        data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        access_token = data.get('access_token')

        if not access_token:
            flash('Failed to get access token', 'error')
            return redirect(url_for('index'))

        # Save token
        save_access_token(access_token, request_token)

        # Get user profile
        kite.set_access_token(access_token)
        profile = kite.profile()
        session['user_name'] = profile.get('user_name', 'User')
        session['user_id'] = profile.get('user_id', '')

        logger.info(f"Login successful for user: {session['user_name']}")
        flash('Login successful!', 'success')

        # Start all tickers after browser login too
        _start_all_tickers()

        return redirect(url_for('backtest_page'))

    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        flash(f'Login failed: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/api/auth/auto-login', methods=['POST'])
def api_auto_login():
    """One-click TOTP auto-login — no browser redirect needed. Starts all tickers."""
    try:
        from services.kite_auth import ensure_authenticated
        if ensure_authenticated():
            try:
                kite = get_kite()
                profile = kite.profile()
                session['user_name'] = profile.get('user_name', 'User')
                session['user_id'] = profile.get('user_id', '')
            except Exception:
                session['user_name'] = 'Authenticated'
                session['user_id'] = ''

            # Auto-start Maruthi ticker after successful login
            _start_all_tickers()

            return jsonify({'status': 'success', 'message': f'Logged in as {session.get("user_name", "User")}. All tickers started.'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'TOTP auto-login failed. Check KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET env vars.'}), 401
    except Exception as e:
        logger.error(f"Auto-login error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _start_all_tickers():
    """Start all strategy tickers/connections after authentication."""
    # Maruthi ticker — DISABLED. Strategy is paused (see memory: 9 known bugs).
    # Leaving this branch in place, gated on MARUTHI_DEFAULTS['enabled'], so the
    # ticker only starts if the strategy is re-enabled. When disabled, NAS owns
    # the process's KiteTicker singleton.
    if MARUTHI_DEFAULTS.get('enabled', False):
        try:
            from services.maruthi_ticker import get_maruthi_ticker
            ticker = get_maruthi_ticker(MARUTHI_DEFAULTS)
            if not ticker.is_connected:
                ticker.restart()
                logger.info("[Auth] Maruthi ticker started after login")
            else:
                logger.info("[Auth] Maruthi ticker already connected")
        except Exception as e:
            logger.warning(f"[Auth] Maruthi ticker start failed: {e}")
    else:
        logger.info("[Auth] Maruthi ticker skipped — strategy disabled")

    # NAS ticker (shared across OTM, ATM, ATM2, ATM4, and 916 variants) —
    # now owns its own KiteTicker WebSocket, no longer depends on Maruthi.
    try:
        from services.nas_ticker import get_nas_ticker
        nas_ticker = get_nas_ticker(NAS_DEFAULTS)
        if not nas_ticker.is_connected:
            nas_ticker.restart()
            logger.info("[Auth] NAS ticker started after login")
        else:
            logger.info("[Auth] NAS ticker already connected")
    except Exception as e:
        logger.warning(f"[Auth] NAS ticker start failed: {e}")


def _bootstrap_tickers_on_startup():
    """On gunicorn boot, if access token is already valid, start tickers
    immediately. This covers `systemctl restart` where the token was cached
    from a prior login — without this, tickers only start on the next TOTP
    login flow."""
    try:
        from services.kite_service import get_access_token
        if not get_access_token():
            logger.info("[Bootstrap] No cached access token — tickers will start after next login")
            return
        # Verify token is actually valid by calling profile()
        try:
            kite = get_kite()
            kite.profile()
        except Exception as e:
            logger.info(f"[Bootstrap] Cached token invalid ({e}) — skipping ticker start")
            return
        logger.info("[Bootstrap] Valid access token found — starting tickers")
        _start_all_tickers()
    except Exception as e:
        logger.warning(f"[Bootstrap] Ticker bootstrap failed: {e}")


# Fire once at module load, in a background thread so gunicorn boot isn't blocked
import threading as _bootstrap_threading
_bootstrap_threading.Thread(target=_bootstrap_tickers_on_startup, daemon=True).start()


@app.route('/api/auth/status')
def api_auth_status():
    """Check if Kite is authenticated."""
    try:
        from kiteconnect import KiteConnect as KC
        token = get_access_token()
        if token:
            kc = KC(api_key=KITE_API_KEY)
            kc.set_access_token(token)
            profile = kc.profile()
            return jsonify({
                'authenticated': True,
                'user_name': profile.get('user_name', ''),
                'user_id': profile.get('user_id', ''),
            })
    except Exception:
        pass
    return jsonify({'authenticated': False})


@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.clear()
    save_access_token('')
    flash('Logged out successfully', 'info')
    return redirect(url_for('index'))


# =============================================================================
# Main Pages
# =============================================================================

@app.route('/backtest')
@login_required
def backtest_page():
    """Main backtest configuration page"""
    return render_template(
        'backtest.html',
        symbols=NIFTY_50,
        top_10=TOP_10_LIQUID,
        lot_sizes=FNO_LOT_SIZES,
        strike_methods=STRIKE_METHODS,
        exit_rules=EXIT_RULES,
        user_name=session.get('user_name', 'User')
    )


@app.route('/backtest/adaptive')
@login_required
def adaptive_backtest_page():
    """Adaptive IV Percentile-based backtest configuration page"""
    # IV regime configuration for display
    iv_regimes = [
        {'name': 'LOW', 'iv_range': '< 25%', 'target_delta': 0.35, 'approx_otm': '3-4%'},
        {'name': 'NORMAL', 'iv_range': '25-50%', 'target_delta': 0.30, 'approx_otm': '4-5%'},
        {'name': 'ELEVATED', 'iv_range': '50-75%', 'target_delta': 0.25, 'approx_otm': '5-7%'},
        {'name': 'HIGH', 'iv_range': '> 75%', 'target_delta': 0.20, 'approx_otm': '7-10%'},
    ]

    return render_template(
        'adaptive_backtest.html',
        symbols=NIFTY_50,
        top_10=TOP_10_LIQUID,
        lot_sizes=FNO_LOT_SIZES,
        exit_rules=EXIT_RULES,
        iv_regimes=iv_regimes,
        user_name=session.get('user_name', 'User')
    )


@app.route('/data-management')
@login_required
def data_management():
    """Data download and management page"""
    dm = get_data_manager()
    summary = dm.get_database_summary()
    status = dm.get_download_status()

    return render_template(
        'data_management.html',
        summary=summary,
        download_status=status.to_dict('records') if len(status) > 0 else [],
        symbols=NIFTY_50,
        user_name=session.get('user_name', 'User')
    )


@app.route('/results/<int:backtest_id>')
@login_required
def results_page(backtest_id: int):
    """Display results for a specific backtest"""
    db = get_backtest_db()
    backtest = db.get_backtest_run(backtest_id)

    if not backtest:
        flash('Backtest not found', 'error')
        return redirect(url_for('backtest_page'))

    trades = db.get_trades(backtest_id)
    equity = db.get_equity_curve(backtest_id)

    return render_template(
        'results.html',
        backtest=backtest,
        trades=trades.to_dict('records') if len(trades) > 0 else [],
        equity=equity.reset_index().to_dict('records') if len(equity) > 0 else [],
        user_name=session.get('user_name', 'User')
    )


@app.route('/positions')
@login_required
def positions_page():
    """Zerodha-style positions page with grouping by instrument/expiry."""
    return render_template(
        'positions.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/api/positions')
@login_required
def api_get_positions():
    """Fetch live positions from Kite API (net + day)."""
    try:
        kite = get_kite()
        positions = kite.positions()
        net = positions.get('net', [])
        day = positions.get('day', [])

        # Enrich with instrument type parsing
        for p in net + day:
            ts = (p.get('tradingsymbol') or '').upper()
            if not p.get('instrument_type'):
                if 'FUT' in ts:
                    p['instrument_type'] = 'FUT'
                elif ts and any(ts.endswith(x) for x in ('CE', 'PE')):
                    p['instrument_type'] = 'CE' if ts.endswith('CE') else 'PE'
                else:
                    p['instrument_type'] = 'EQ'

        return jsonify({'net': net, 'day': day})
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/holdings')
@login_required
def holdings_page():
    """Holdings dashboard with flip cards and expanded view"""
    return render_template(
        'holdings.html',
        user_name=session.get('user_name', 'User')
    )


@app.route('/holdings/detail/<symbol>')
@login_required
def stock_detail_page(symbol: str):
    """Full-screen detailed stock view with comprehensive data"""
    return render_template(
        'stock_detail.html',
        symbol=symbol,
        user_name=session.get('user_name', 'User')
    )


# =============================================================================
# API Routes - Holdings
# =============================================================================

@app.route('/api/holdings')
@login_required
def api_get_holdings():
    """Get user holdings from Zerodha with P/L calculations"""
    try:
        holdings = get_holdings()
        summary = get_portfolio_summary(holdings)
        return jsonify({
            'holdings': holdings,
            'summary': summary
        })
    except Exception as e:
        logger.error(f"Error fetching holdings: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fundamentals/<symbol>')
@login_required
def api_get_fundamentals(symbol: str):
    """Get fundamental data for a stock from Yahoo Finance"""
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        fundamentals = get_fundamentals(symbol, force_refresh=force_refresh)
        return jsonify(fundamentals)
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/historical/<symbol>')
@login_required
def api_get_historical(symbol: str):
    """Get historical prices for sparkline charts"""
    try:
        period = request.args.get('period', '1y')
        interval = request.args.get('interval', None)  # e.g., '5m', '15m' for intraday
        prices = get_historical_prices(symbol, period=period, interval=interval)
        return jsonify({'symbol': symbol, 'prices': prices})
    except Exception as e:
        logger.error(f"Error fetching historical prices for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trading-data', methods=['POST'])
@login_required
def api_get_trading_data():
    """Get trading data (CPR, EMA, today's %) for multiple symbols"""
    try:
        from services.holdings_service import get_trading_data
        data = request.get_json()
        symbols = data.get('symbols', [])
        if not symbols:
            return jsonify({'trading_data': {}})

        trading_data = get_trading_data(symbols)
        return jsonify({'trading_data': trading_data})
    except Exception as e:
        logger.error(f"Error fetching trading data: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# API Routes - Claude Chat (Portfolio Research)
# =============================================================================

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    """Send a message to Claude for portfolio research"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Use user_id as session identifier for conversation history
        session_id = session.get('user_id', 'default')

        # Get response from Claude with portfolio tools
        response = portfolio_chat(session_id, message)

        return jsonify({
            'response': response,
            'session_id': session_id
        })

    except ValueError as e:
        # API key not configured
        logger.error(f"Chat configuration error: {e}")
        return jsonify({
            'error': 'Claude API not configured. Please set ANTHROPIC_API_KEY environment variable.'
        }), 503

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/suggestions')
@login_required
def api_chat_suggestions():
    """Get suggested questions for the chat interface"""
    try:
        suggestions = get_suggested_questions()
        return jsonify({'suggestions': suggestions})
    except Exception as e:
        logger.error(f"Error getting chat suggestions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/clear', methods=['POST'])
@login_required
def api_chat_clear():
    """Clear chat history for current session"""
    try:
        session_id = session.get('user_id', 'default')
        clear_chat_history(session_id)
        return jsonify({'status': 'ok', 'message': 'Chat history cleared'})
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# API Routes - Backtest
# =============================================================================

@app.route('/api/backtest/run', methods=['POST'])
@login_required
def api_run_backtest():
    """Start a new backtest"""
    try:
        data = request.get_json()

        # Validate required fields
        symbols = data.get('symbols', [])
        if not symbols:
            return jsonify({'error': 'No symbols selected'}), 400

        start_date = datetime.strptime(data.get('start_date'), '%Y-%m-%d')
        end_date = datetime.strptime(data.get('end_date'), '%Y-%m-%d')

        if end_date <= start_date:
            return jsonify({'error': 'End date must be after start date'}), 400

        strike_method = data.get('strike_method', 'DELTA_30')
        exit_strategy = data.get('exit_strategy', 'HOLD_TO_EXPIRY')
        initial_capital = float(data.get('initial_capital', 1000000))

        # New exit parameters
        profit_target_pct = float(data.get('profit_target_pct', 50))
        stop_loss_multiple = float(data.get('stop_loss_multiple', 2.0))

        # Stop-loss adjustment (roll-up) option
        allow_sl_adjustment = data.get('allow_sl_adjustment', False)

        # Trend filter options
        use_trend_filter = data.get('use_trend_filter', False)  # DEPRECATED
        trend_filter_mode = data.get('trend_filter_mode', 'NONE')  # Multi-timeframe EMA filter

        # RSI filter options
        use_rsi_filter = data.get('use_rsi_filter', False)
        rsi_period = int(data.get('rsi_period', 14))
        rsi_min = float(data.get('rsi_min', 40.0))
        rsi_max = float(data.get('rsi_max', 70.0))

        # Stochastic filter option
        use_stochastic_filter = data.get('use_stochastic_filter', False)
        stochastic_k_period = int(data.get('stochastic_k_period', 14))
        stochastic_d_period = int(data.get('stochastic_d_period', 3))
        stochastic_smoothing = int(data.get('stochastic_smoothing', 3))
        stochastic_overbought = float(data.get('stochastic_overbought', 70.0))

        # Advanced exit strategies - DTE-based exit
        use_dte_exit = data.get('use_dte_exit', False)
        dte_exit_threshold = int(data.get('dte_exit_threshold', 7))

        # Advanced exit strategies - Trailing stop
        use_trailing_stop = data.get('use_trailing_stop', False)
        trailing_stop_activation = float(data.get('trailing_stop_activation', 25.0))
        trailing_stop_distance = float(data.get('trailing_stop_distance', 15.0))

        # Supertrend filter
        use_supertrend_filter = data.get('use_supertrend_filter', False)
        supertrend_period = int(data.get('supertrend_period', 10))
        supertrend_multiplier = float(data.get('supertrend_multiplier', 3.0))

        # VWAP filter
        use_vwap_filter = data.get('use_vwap_filter', False)
        vwap_mode = data.get('vwap_mode', 'ABOVE')
        vwap_period = int(data.get('vwap_period', 1))

        # Bollinger Bands filter
        use_bollinger_filter = data.get('use_bollinger_filter', False)
        bollinger_period = int(data.get('bollinger_period', 20))
        bollinger_std = float(data.get('bollinger_std', 2.0))

        # ADX filter (trend strength)
        use_adx_filter = data.get('use_adx_filter', False)
        adx_period = int(data.get('adx_period', 14))
        adx_threshold = float(data.get('adx_threshold', 25.0))
        adx_require_bullish = data.get('adx_require_bullish', True)

        # MACD filter (momentum)
        use_macd_filter = data.get('use_macd_filter', False)
        macd_fast = int(data.get('macd_fast', 12))
        macd_slow = int(data.get('macd_slow', 26))
        macd_signal = int(data.get('macd_signal', 9))
        macd_mode = data.get('macd_mode', 'BULLISH')

        # Williams %R filter (momentum oscillator)
        use_williams_filter = data.get('use_williams_filter', False)
        williams_period = int(data.get('williams_period', 14))
        williams_overbought = float(data.get('williams_overbought', -20.0))
        williams_oversold = float(data.get('williams_oversold', -80.0))

        # Create task ID
        task_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize task status
        task_status[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Initializing backtest...',
            'result': None
        }

        # Schedule backtest execution
        scheduler.add_job(
            _execute_backtest,
            args=[task_id, symbols, start_date, end_date,
                  strike_method, exit_strategy, initial_capital,
                  profit_target_pct, stop_loss_multiple, allow_sl_adjustment,
                  use_trend_filter, trend_filter_mode,
                  use_rsi_filter, rsi_period, rsi_min, rsi_max,
                  use_stochastic_filter,
                  stochastic_k_period, stochastic_d_period,
                  stochastic_smoothing, stochastic_overbought,
                  use_dte_exit, dte_exit_threshold,
                  use_trailing_stop, trailing_stop_activation, trailing_stop_distance,
                  use_supertrend_filter, supertrend_period, supertrend_multiplier,
                  use_vwap_filter, vwap_mode, vwap_period,
                  use_bollinger_filter, bollinger_period, bollinger_std,
                  use_adx_filter, adx_period, adx_threshold, adx_require_bullish,
                  use_macd_filter, macd_fast, macd_slow, macd_signal, macd_mode,
                  use_williams_filter, williams_period, williams_overbought, williams_oversold],
            id=task_id
        )

        return jsonify({
            'task_id': task_id,
            'status': 'started',
            'message': 'Backtest started'
        })

    except Exception as e:
        logger.error(f"Error starting backtest: {e}")
        return jsonify({'error': str(e)}), 500


def _execute_backtest(
    task_id: str,
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    strike_method: str,
    exit_strategy: str,
    initial_capital: float,
    profit_target_pct: float = 50.0,
    stop_loss_multiple: float = 2.0,
    allow_sl_adjustment: bool = False,
    use_trend_filter: bool = False,
    trend_filter_mode: str = "NONE",
    use_rsi_filter: bool = False,
    rsi_period: int = 14,
    rsi_min: float = 40.0,
    rsi_max: float = 70.0,
    use_stochastic_filter: bool = False,
    stochastic_k_period: int = 14,
    stochastic_d_period: int = 3,
    stochastic_smoothing: int = 3,
    stochastic_overbought: float = 70.0,
    use_dte_exit: bool = False,
    dte_exit_threshold: int = 7,
    use_trailing_stop: bool = False,
    trailing_stop_activation: float = 25.0,
    trailing_stop_distance: float = 15.0,
    use_supertrend_filter: bool = False,
    supertrend_period: int = 10,
    supertrend_multiplier: float = 3.0,
    use_vwap_filter: bool = False,
    vwap_mode: str = "ABOVE",
    vwap_period: int = 1,
    use_bollinger_filter: bool = False,
    bollinger_period: int = 20,
    bollinger_std: float = 2.0,
    use_adx_filter: bool = False,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    adx_require_bullish: bool = True,
    use_macd_filter: bool = False,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    macd_mode: str = "BULLISH",
    use_williams_filter: bool = False,
    williams_period: int = 14,
    williams_overbought: float = -20.0,
    williams_oversold: float = -80.0
):
    """Execute backtest in background"""
    try:
        def progress_callback(pct, msg):
            task_status[task_id]['progress'] = pct
            task_status[task_id]['message'] = msg

        task_status[task_id]['message'] = 'Loading market data...'

        # Load market data
        dm = get_data_manager()
        stock_data = {}

        for symbol in symbols:
            try:
                df = dm.load_data(symbol, 'day', start_date, end_date)
                stock_data[symbol] = df
            except Exception as e:
                logger.warning(f"Could not load data for {symbol}: {e}")

        if not stock_data:
            raise ValueError("No market data available for selected symbols")

        task_status[task_id]['message'] = 'Running backtest...'

        # Run backtest
        config = BacktestConfig(
            symbols=list(stock_data.keys()),
            start_date=start_date,
            end_date=end_date,
            strike_method=StrikeMethod(strike_method),
            exit_strategy=ExitStrategy(exit_strategy),
            initial_capital=initial_capital,
            profit_target_pct=profit_target_pct,
            stop_loss_multiple=stop_loss_multiple,
            allow_sl_adjustment=allow_sl_adjustment,
            use_trend_filter=use_trend_filter,
            trend_filter_mode=trend_filter_mode,
            use_rsi_filter=use_rsi_filter,
            rsi_period=rsi_period,
            rsi_min=rsi_min,
            rsi_max=rsi_max,
            use_stochastic_filter=use_stochastic_filter,
            stochastic_k_period=stochastic_k_period,
            stochastic_d_period=stochastic_d_period,
            stochastic_smoothing=stochastic_smoothing,
            stochastic_overbought=stochastic_overbought,
            use_dte_exit=use_dte_exit,
            dte_exit_threshold=dte_exit_threshold,
            use_trailing_stop=use_trailing_stop,
            trailing_stop_activation=trailing_stop_activation,
            trailing_stop_distance=trailing_stop_distance,
            use_supertrend_filter=use_supertrend_filter,
            supertrend_period=supertrend_period,
            supertrend_multiplier=supertrend_multiplier,
            use_vwap_filter=use_vwap_filter,
            vwap_mode=vwap_mode,
            vwap_period=vwap_period,
            use_bollinger_filter=use_bollinger_filter,
            bollinger_period=bollinger_period,
            bollinger_std=bollinger_std,
            use_adx_filter=use_adx_filter,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            adx_require_bullish=adx_require_bullish,
            use_macd_filter=use_macd_filter,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            macd_mode=macd_mode,
            use_williams_filter=use_williams_filter,
            williams_period=williams_period,
            williams_overbought=williams_overbought,
            williams_oversold=williams_oversold
        )

        engine = CoveredCallEngine(config)
        results = engine.run_backtest(stock_data, progress_callback)

        # Save to database
        db = get_backtest_db()
        backtest_id = db.create_backtest_run(
            name=f"Backtest {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            config=results['config'],
            symbols=symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            strike_method=strike_method,
            exit_strategy=exit_strategy
        )

        # Save trades
        if len(results['trades']) > 0:
            trades_list = results['trades'].to_dict('records')
            db.add_trades_batch(backtest_id, trades_list)

        # Save equity curve
        equity_points = []
        for date, value in results['equity_curve'].items():
            equity_points.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': value
            })
        if equity_points:
            db.add_equity_points(backtest_id, equity_points)

        # Update with final metrics
        db.update_backtest_metrics(backtest_id, results['metrics'])

        task_status[task_id]['status'] = 'completed'
        task_status[task_id]['progress'] = 100
        task_status[task_id]['message'] = 'Backtest completed!'
        task_status[task_id]['result'] = {
            'backtest_id': backtest_id,
            'metrics': results['metrics']
        }

    except Exception as e:
        logger.error(f"Backtest execution error: {e}")
        task_status[task_id]['status'] = 'failed'
        task_status[task_id]['message'] = str(e)


@app.route('/api/backtest/status/<task_id>')
@login_required
def api_backtest_status(task_id: str):
    """Get status of a running backtest"""
    if task_id not in task_status:
        return jsonify({'error': 'Task not found'}), 404

    return jsonify(task_status[task_id])


@app.route('/api/backtest/history')
@login_required
def api_backtest_history():
    """Get list of recent backtests"""
    db = get_backtest_db()
    backtests = db.get_recent_backtests(limit=20)
    return jsonify(backtests)


@app.route('/api/backtest/<int:backtest_id>')
@login_required
def api_get_backtest(backtest_id: int):
    """Get full backtest results"""
    db = get_backtest_db()
    backtest = db.get_backtest_run(backtest_id)

    if not backtest:
        return jsonify({'error': 'Backtest not found'}), 404

    trades = db.get_trades(backtest_id)
    equity = db.get_equity_curve(backtest_id)

    return jsonify({
        'backtest': backtest,
        'trades': trades.to_dict('records') if len(trades) > 0 else [],
        'equity': equity.reset_index().to_dict('records') if len(equity) > 0 else []
    })


# =============================================================================
# API Routes - Data Management
# =============================================================================

@app.route('/api/data/download', methods=['POST'])
@login_required
def api_download_data():
    """Start data download"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        timeframe = data.get('timeframe', 'day')

        if not symbols:
            return jsonify({'error': 'No symbols selected'}), 400

        task_id = f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        task_status[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Starting download...',
            'result': None
        }

        # Get Kite connection
        kite = get_kite()
        dm = get_data_manager(kite)

        # Calculate date range (2 years)
        from_date = datetime.now() - timedelta(days=730)
        to_date = datetime.now()

        # Schedule download
        scheduler.add_job(
            _execute_download,
            args=[task_id, dm, symbols, timeframe, from_date, to_date],
            id=task_id
        )

        return jsonify({
            'task_id': task_id,
            'status': 'started',
            'message': f'Downloading {len(symbols)} symbols'
        })

    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': str(e)}), 500


def _execute_download(task_id, dm, symbols, timeframe, from_date, to_date):
    """Execute data download in background"""
    try:
        def progress_callback(idx, total, symbol, status):
            pct = (idx / total) * 100
            task_status[task_id]['progress'] = pct
            task_status[task_id]['message'] = f'{symbol}: {status}'

        success, failed, errors = dm.download_data(
            symbols=symbols,
            timeframe=timeframe,
            from_date=from_date,
            to_date=to_date,
            progress_callback=progress_callback
        )

        task_status[task_id]['status'] = 'completed'
        task_status[task_id]['progress'] = 100
        task_status[task_id]['message'] = f'Download complete: {success} success, {failed} failed'
        task_status[task_id]['result'] = {
            'success': success,
            'failed': failed,
            'errors': errors
        }

    except Exception as e:
        logger.error(f"Download execution error: {e}")
        task_status[task_id]['status'] = 'failed'
        task_status[task_id]['message'] = str(e)


@app.route('/api/data/status/<task_id>')
@login_required
def api_download_status(task_id: str):
    """Get status of a running download"""
    if task_id not in task_status:
        return jsonify({'error': 'Task not found'}), 404

    return jsonify(task_status[task_id])


@app.route('/api/data/summary')
@login_required
def api_data_summary():
    """Get database summary"""
    dm = get_data_manager()
    summary = dm.get_database_summary()
    return jsonify(summary)


@app.route('/api/data/symbols')
@login_required
def api_available_symbols():
    """Get list of symbols with data"""
    dm = get_data_manager()
    symbols = dm.get_available_symbols('day')
    return jsonify({'symbols': symbols})


# =============================================================================
# CPR Strategy Backtest Routes
# =============================================================================

@app.route('/backtest/cpr')
@login_required
def cpr_backtest_page():
    """CPR-based covered call backtest configuration page"""
    return render_template(
        'cpr_backtest.html',
        symbols=NIFTY_50,
        top_10=TOP_10_LIQUID,
        lot_sizes=FNO_LOT_SIZES,
        user_name=session.get('user_name', 'User')
    )


@app.route('/api/cpr-backtest/run', methods=['POST'])
@login_required
def api_run_cpr_backtest():
    """Start a CPR-based covered call backtest"""
    try:
        data = request.get_json()

        # Validate required fields
        symbols = data.get('symbols', [])
        if not symbols:
            return jsonify({'error': 'No symbols selected'}), 400

        start_date = datetime.strptime(data.get('start_date'), '%Y-%m-%d')
        end_date = datetime.strptime(data.get('end_date'), '%Y-%m-%d')

        if end_date <= start_date:
            return jsonify({'error': 'End date must be after start date'}), 400

        # CPR-specific parameters
        narrow_cpr_threshold = float(data.get('narrow_cpr_threshold', 0.5))
        otm_strike_pct = float(data.get('otm_strike_pct', 5.0))
        dte_min = int(data.get('dte_min', 30))
        dte_max = int(data.get('dte_max', 35))
        enable_premium_rollout = data.get('enable_premium_rollout', True)
        premium_double_threshold = float(data.get('premium_double_threshold', 2.0))
        premium_erosion_target = float(data.get('premium_erosion_target', 75.0))
        dte_exit_threshold = int(data.get('dte_exit_threshold', 10))
        enable_r1_exit = data.get('enable_r1_exit', True)
        use_closer_r1 = data.get('use_closer_r1', True)
        initial_capital = float(data.get('initial_capital', 1000000))

        # Create task ID
        task_id = f"cpr_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize task status
        task_status[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Initializing CPR backtest...',
            'result': None
        }

        # Schedule backtest execution
        scheduler.add_job(
            _execute_cpr_backtest,
            args=[task_id, symbols, start_date, end_date,
                  narrow_cpr_threshold, otm_strike_pct, dte_min, dte_max,
                  enable_premium_rollout, premium_double_threshold,
                  premium_erosion_target, dte_exit_threshold,
                  enable_r1_exit, use_closer_r1, initial_capital],
            id=task_id
        )

        return jsonify({
            'task_id': task_id,
            'status': 'started',
            'message': 'CPR Backtest started'
        })

    except Exception as e:
        logger.error(f"Error starting CPR backtest: {e}")
        return jsonify({'error': str(e)}), 500


def _execute_cpr_backtest(
    task_id: str,
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    narrow_cpr_threshold: float,
    otm_strike_pct: float,
    dte_min: int,
    dte_max: int,
    enable_premium_rollout: bool,
    premium_double_threshold: float,
    premium_erosion_target: float,
    dte_exit_threshold: int,
    enable_r1_exit: bool,
    use_closer_r1: bool,
    initial_capital: float
):
    """Execute CPR backtest in background"""
    try:
        def progress_callback(pct, msg):
            task_status[task_id]['progress'] = pct
            task_status[task_id]['message'] = msg

        task_status[task_id]['message'] = 'Loading market data...'

        # Load daily market data
        dm = get_data_manager()
        stock_data = {}

        for symbol in symbols:
            try:
                df = dm.load_data(symbol, 'day', start_date, end_date)
                if df is not None and not df.empty:
                    stock_data[symbol] = df
            except Exception as e:
                logger.warning(f"Could not load data for {symbol}: {e}")

        if not stock_data:
            raise ValueError("No market data available for selected symbols")

        task_status[task_id]['message'] = 'Loading intraday data...'

        # Load intraday data from bridge
        intraday_bridge = get_intraday_bridge()
        intraday_data = {}

        for symbol in stock_data.keys():
            try:
                df = intraday_bridge.load_30min_data(
                    symbol,
                    start_date - timedelta(days=14),
                    end_date
                )
                if df is not None and not df.empty:
                    intraday_data[symbol] = df
            except Exception as e:
                logger.warning(f"Could not load intraday data for {symbol}: {e}")

        task_status[task_id]['message'] = 'Running CPR backtest...'

        # Create config
        config = CPRBacktestConfig(
            symbols=list(stock_data.keys()),
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            narrow_cpr_threshold=narrow_cpr_threshold,
            otm_strike_pct=otm_strike_pct,
            dte_min=dte_min,
            dte_max=dte_max,
            enable_premium_rollout=enable_premium_rollout,
            premium_double_threshold=premium_double_threshold,
            premium_erosion_target=premium_erosion_target,
            dte_exit_threshold=dte_exit_threshold,
            enable_r1_exit=enable_r1_exit,
            use_closer_r1=use_closer_r1
        )

        # Run backtest
        engine = CPRCoveredCallEngine(config)
        results = engine.run_backtest(stock_data, intraday_data, progress_callback)

        task_status[task_id]['status'] = 'completed'
        task_status[task_id]['progress'] = 100
        task_status[task_id]['message'] = 'CPR Backtest completed!'
        task_status[task_id]['result'] = results

    except Exception as e:
        logger.error(f"CPR Backtest execution error: {e}")
        import traceback
        traceback.print_exc()
        task_status[task_id]['status'] = 'error'
        task_status[task_id]['message'] = str(e)
        task_status[task_id]['error'] = str(e)


@app.route('/api/cpr-backtest/status/<task_id>')
@login_required
def api_cpr_backtest_status(task_id: str):
    """Get status of a running CPR backtest"""
    if task_id not in task_status:
        return jsonify({'error': 'Task not found'}), 404

    return jsonify(task_status[task_id])


@app.route('/api/cpr-backtest/optimize', methods=['POST'])
@login_required
def api_run_cpr_optimization():
    """Start CPR strategy optimization in background"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])

        if not symbols:
            return jsonify({'error': 'No symbols selected'}), 400

        start_date = datetime.strptime(data.get('start_date'), '%Y-%m-%d')
        end_date = datetime.strptime(data.get('end_date'), '%Y-%m-%d')

        task_id = f"cpr_optimize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        task_status[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Initializing optimization...',
            'result': None
        }

        scheduler.add_job(
            _execute_cpr_optimization,
            args=[task_id, symbols, start_date, end_date],
            id=task_id
        )

        return jsonify({'task_id': task_id, 'status': 'started'})

    except Exception as e:
        logger.error(f"Error starting CPR optimization: {e}")
        return jsonify({'error': str(e)}), 500


def _execute_cpr_optimization(task_id: str, symbols: list, start_date: datetime, end_date: datetime):
    """Execute CPR optimization in background"""
    try:
        from services.cpr_strategy_optimizer import CPRStrategyOptimizer

        def progress_callback(pct, msg):
            task_status[task_id]['progress'] = pct
            task_status[task_id]['message'] = msg

        # Load data
        dm = get_data_manager()
        stock_data = {}
        for symbol in symbols:
            try:
                df = dm.load_data(symbol, 'day', start_date, end_date)
                if df is not None and not df.empty:
                    stock_data[symbol] = df
            except:
                pass

        intraday_bridge = get_intraday_bridge()
        intraday_data = {}
        for symbol in stock_data.keys():
            try:
                df = intraday_bridge.load_30min_data(symbol, start_date - timedelta(days=14), end_date)
                if df is not None and not df.empty:
                    intraday_data[symbol] = df
            except:
                pass

        # Run optimization
        optimizer = CPRStrategyOptimizer(
            symbols=list(stock_data.keys()),
            start_date=start_date,
            end_date=end_date,
            stock_data=stock_data,
            intraday_data=intraday_data
        )

        results = optimizer.run_optimization(progress_callback=progress_callback)

        task_status[task_id]['status'] = 'completed'
        task_status[task_id]['progress'] = 100
        task_status[task_id]['message'] = 'Optimization completed!'
        task_status[task_id]['result'] = results
        task_status[task_id]['best_sharpe'] = results.get('best_sharpe', 0)

    except Exception as e:
        logger.error(f"CPR optimization error: {e}")
        task_status[task_id]['status'] = 'error'
        task_status[task_id]['message'] = str(e)


@app.route('/api/cpr-backtest/optimization-results/<task_id>')
@login_required
def api_get_cpr_optimization_results(task_id: str):
    """Get CPR optimization results"""
    if task_id not in task_status:
        return jsonify({'error': 'Task not found'}), 404

    return jsonify(task_status[task_id])


# =============================================================================
# Live Ticker WebSocket Routes
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('ticker_status', {'status': 'connected', 'message': 'WebSocket connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client WebSocket disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('subscribe_ticker')
def handle_subscribe_ticker(data):
    """Subscribe to live ticker for given symbols"""
    global _ticker_service

    symbols = data.get('symbols', [])
    if not symbols:
        emit('ticker_error', {'error': 'No symbols provided'})
        return

    try:
        from services.kite_ticker_service import get_ticker_service, KiteTickerService

        access_token = get_access_token()
        if not access_token:
            emit('ticker_error', {'error': 'Not authenticated. Please login first.'})
            return

        # Get or create ticker service
        _ticker_service = get_ticker_service(access_token)

        if not _ticker_service:
            emit('ticker_error', {'error': 'Could not initialize ticker service'})
            return

        # Check if we need to update instrument tokens
        if not _ticker_service.symbol_to_token:
            logger.info("Updating instrument tokens...")
            kite = get_kite()
            _ticker_service.update_instruments(kite)

        # Set up tick callback to broadcast via SocketIO
        def on_tick(tick_data):
            socketio.emit('price_tick', tick_data)

        _ticker_service.set_callback(on_tick)

        # Subscribe to symbols
        subscribed = _ticker_service.subscribe(symbols)

        # Start ticker if not already running
        if not _ticker_service.is_running():
            _ticker_service.start(threaded=True)

        emit('ticker_subscribed', {
            'subscribed': subscribed,
            'total': len(symbols),
            'missing': [s for s in symbols if s not in subscribed]
        })

        logger.info(f"Subscribed to {len(subscribed)} symbols: {subscribed}")

    except Exception as e:
        logger.error(f"Error subscribing to ticker: {e}")
        emit('ticker_error', {'error': str(e)})


@socketio.on('unsubscribe_ticker')
def handle_unsubscribe_ticker(data):
    """Unsubscribe from live ticker for given symbols"""
    global _ticker_service

    symbols = data.get('symbols', [])
    if not symbols or not _ticker_service:
        return

    try:
        _ticker_service.unsubscribe(symbols)
        emit('ticker_unsubscribed', {'symbols': symbols})
    except Exception as e:
        logger.error(f"Error unsubscribing from ticker: {e}")


@socketio.on('subscribe_position_tokens')
def handle_subscribe_position_tokens(data):
    """Subscribe to live ticks by instrument token (for positions page).

    Expects: { tokens: [{token: int, tradingsymbol: str}, ...] }
    Reuses MaruthiTicker's KiteTicker connection to avoid hitting the 3-connection limit.
    """
    items = data.get('tokens', [])
    if not items:
        emit('pos_error', {'error': 'No tokens provided'})
        return

    try:
        from services.maruthi_ticker import get_maruthi_ticker
        ticker = get_maruthi_ticker()

        if not ticker.is_connected or not ticker.kws:
            emit('pos_error', {'error': 'Ticker not connected. Start Maruthi ticker first.'})
            return

        token_ids = [int(t['token']) for t in items]
        tsym_map = {int(t['token']): t.get('tradingsymbol', '') for t in items}

        # Store mapping on ticker so we can resolve in tick handler
        if not hasattr(ticker, '_pos_token_map'):
            ticker._pos_token_map = {}
        ticker._pos_token_map.update(tsym_map)

        # Subscribe via the existing WebSocket
        ticker.kws.subscribe(token_ids)
        ticker.kws.set_mode(ticker.kws.MODE_LTP, token_ids)

        logger.info(f"[Positions] Subscribed to {len(token_ids)} tokens for live ticks")
        emit('pos_subscribed', {'count': len(token_ids)})

    except Exception as e:
        logger.error(f"[Positions] Token subscription error: {e}")
        emit('pos_error', {'error': str(e)})


@socketio.on('unsubscribe_position_tokens')
def handle_unsubscribe_position_tokens(data):
    """Unsubscribe position tokens when leaving the page."""
    token_ids = data.get('tokens', [])
    if not token_ids:
        return
    try:
        from services.maruthi_ticker import get_maruthi_ticker
        ticker = get_maruthi_ticker()
        if ticker.is_connected and ticker.kws:
            ticker.kws.unsubscribe(token_ids)
            # Clean up mapping
            if hasattr(ticker, '_pos_token_map'):
                for t in token_ids:
                    ticker._pos_token_map.pop(int(t), None)
            logger.info(f"[Positions] Unsubscribed {len(token_ids)} tokens")
    except Exception as e:
        logger.error(f"[Positions] Unsubscribe error: {e}")


@app.route('/api/ticker/start', methods=['POST'])
@login_required
def api_start_ticker():
    """Start the live ticker service"""
    global _ticker_service

    try:
        from services.kite_ticker_service import get_ticker_service

        access_token = get_access_token()
        if not access_token:
            return jsonify({'error': 'Not authenticated'}), 401

        _ticker_service = get_ticker_service(access_token)

        if not _ticker_service:
            return jsonify({'error': 'Could not initialize ticker service'}), 500

        # Update instruments if needed
        if not _ticker_service.symbol_to_token:
            kite = get_kite()
            _ticker_service.update_instruments(kite)

        # Set up broadcast callback
        def on_tick(tick_data):
            socketio.emit('price_tick', tick_data)

        _ticker_service.set_callback(on_tick)

        if not _ticker_service.is_running():
            _ticker_service.start(threaded=True)

        return jsonify({
            'status': 'started',
            'instruments_loaded': len(_ticker_service.symbol_to_token)
        })

    except Exception as e:
        logger.error(f"Error starting ticker: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ticker/stop', methods=['POST'])
@login_required
def api_stop_ticker():
    """Stop the live ticker service"""
    global _ticker_service

    try:
        from services.kite_ticker_service import stop_ticker_service

        stop_ticker_service()
        _ticker_service = None

        return jsonify({'status': 'stopped'})

    except Exception as e:
        logger.error(f"Error stopping ticker: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ticker/status')
@login_required
def api_ticker_status():
    """Get ticker service status"""
    global _ticker_service

    if not _ticker_service:
        return jsonify({
            'running': False,
            'connected': False,
            'subscribed_count': 0
        })

    return jsonify({
        'running': True,
        'connected': _ticker_service.is_connected,
        'subscribed_count': len(_ticker_service.subscribed_tokens),
        'instruments_loaded': len(_ticker_service.symbol_to_token)
    })


@app.route('/api/ticker/subscribe', methods=['POST'])
@login_required
def api_subscribe_symbols():
    """Subscribe to additional symbols"""
    global _ticker_service

    if not _ticker_service:
        return jsonify({'error': 'Ticker not started'}), 400

    data = request.get_json()
    symbols = data.get('symbols', [])

    if not symbols:
        return jsonify({'error': 'No symbols provided'}), 400

    subscribed = _ticker_service.subscribe(symbols)

    return jsonify({
        'subscribed': subscribed,
        'total': len(symbols)
    })


# =============================================================================
# MQ Agent Routes
# =============================================================================

@app.route('/agent')
def mq_agent_dashboard():
    """MQ Agent Dashboard - no login required (read-only strategy dashboard)"""
    return render_template(
        'mq_dashboard.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/api/agent/state')
def api_agent_state():
    """Get current agent state: portfolio, regime, last backtest curves."""
    try:
        from services.mq_agent_db import get_agent_db
        db = get_agent_db()

        portfolio = db.get_portfolio_state() or {}

        # Get regime from DB (persists across restarts)
        db_regime = db.get_regime()
        if db_regime:
            regime = {
                'regime': db_regime.get('regime', 'UNKNOWN'),
                'index_close': db_regime.get('index_close', 0),
                'index_200dma': db_regime.get('index_200dma', 0),
                'vix': db_regime.get('vix'),
            }
        else:
            regime = {'regime': 'UNKNOWN', 'index_close': 0, 'index_200dma': 0, 'vix': None}

        # Get last backtest results from task_status cache
        equity_curve = {}
        benchmark_curves = {}
        sector_allocation = {}
        backtest_metrics = {}
        for tid, ts in list(task_status.items()):
            if tid.startswith('mq_backtest_') and ts.get('status') == 'completed':
                result = ts.get('result', {})
                equity_curve = result.get('equity_curve', {})
                benchmark_curves = result.get('benchmark_curves', {})
                sector_allocation = result.get('sector_allocation', {})
                backtest_metrics = result.get('metrics', {})
                break
        if not backtest_metrics:
            try:
                import json
                bt_file = Path('backtest_data') / 'last_backtest.json'
                if bt_file.exists():
                    bt_data = json.loads(bt_file.read_text(encoding='utf-8'))
                    backtest_metrics = bt_data.get('metrics', {})
                    if not sector_allocation:
                        sector_allocation = bt_data.get('sector_allocation', {})
            except Exception:
                pass

        # Get last screening results (from memory or persisted file)
        screening = {}
        for tid, ts in list(task_status.items()):
            if tid.startswith('mq_screen_') and ts.get('status') == 'completed':
                screening = ts.get('result', {})
                break
        if not screening:
            try:
                import json
                screening_file = Path('backtest_data') / 'last_screening.json'
                if screening_file.exists():
                    screening = json.loads(screening_file.read_text(encoding='utf-8'))
            except Exception:
                pass

        return jsonify({
            'portfolio': portfolio,
            'regime': regime,
            'equity_curve': equity_curve,
            'benchmark_curves': benchmark_curves,
            'sector_allocation': sector_allocation,
            'backtest_metrics': backtest_metrics,
            'screening': screening,
        })
    except Exception as e:
        logger.error(f"Agent state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent/signals')
def api_agent_signals():
    """Get active signals from the agent DB."""
    try:
        from services.mq_agent_db import get_agent_db
        db = get_agent_db()
        signals = db.get_active_signals(limit=50)
        return jsonify(signals)
    except Exception as e:
        logger.error(f"Agent signals error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent/runs')
def api_agent_runs():
    """Get recent agent runs and reports."""
    try:
        from services.mq_agent_db import get_agent_db
        db = get_agent_db()
        runs = db.get_recent_runs(limit=20)
        reports = db.get_recent_reports(limit=10)
        return jsonify({'runs': runs, 'reports': reports})
    except Exception as e:
        logger.error(f"Agent runs error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent/backtest', methods=['POST'])
def api_agent_run_backtest():
    """Start an MQ backtest in the background."""
    try:
        task_id = f"mq_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        task_status[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Initializing MQ backtest...',
            'result': None,
        }

        scheduler.add_job(
            _execute_mq_backtest,
            args=[task_id],
            id=task_id,
        )

        return jsonify({'task_id': task_id, 'status': 'started'})
    except Exception as e:
        logger.error(f"MQ backtest start error: {e}")
        return jsonify({'error': str(e)}), 500


def _execute_mq_backtest(task_id: str):
    """Run MQ backtest in background thread."""
    try:
        from services.mq_backtest_agent import BacktestAgent
        from services.mq_backtest_engine import MQBacktestConfig

        def progress_cb(day_idx, total_days, current_date, msg):
            pct = (day_idx / total_days * 100) if total_days else 0
            task_status[task_id]['progress'] = round(pct, 1)
            task_status[task_id]['message'] = msg

        agent = BacktestAgent()
        config = MQBacktestConfig()
        report = agent.run(config=config, progress_callback=progress_cb)

        # Cache results for dashboard display
        backtest_result = {
            'metrics': report.metrics,
            'equity_curve': report.equity_curve,
            'benchmark_curves': report.benchmark_curves,
            'sector_allocation': report.sector_allocation,
            'trade_count': report.trade_count,
        }
        task_status[task_id]['status'] = 'completed'
        task_status[task_id]['progress'] = 100
        task_status[task_id]['message'] = f"CAGR {report.metrics.get('cagr', 0)}%, Sharpe {report.metrics.get('sharpe_ratio', 0)}"
        task_status[task_id]['result'] = backtest_result

        # Persist backtest metrics so they survive restarts
        try:
            import json as json_mod
            bt_file = Path('backtest_data') / 'last_backtest.json'
            bt_file.parent.mkdir(parents=True, exist_ok=True)
            # Save metrics + sector allocation (skip large equity curves)
            bt_file.write_text(json_mod.dumps({
                'metrics': report.metrics,
                'sector_allocation': report.sector_allocation,
                'trade_count': report.trade_count,
                'run_date': datetime.now().isoformat(),
            }, default=str), encoding='utf-8')
        except Exception as pf:
            logger.warning(f"Could not persist backtest results: {pf}")

        # Generate HTML report and save path to DB
        try:
            from services.mq_reporting_agent import ReportingAgent
            from services.mq_agent_db import get_agent_db
            reporter = ReportingAgent()
            report_path = reporter.generate_backtest_report(report)
            db = get_agent_db()
            latest = db.get_latest_run('backtest')
            if latest and report_path:
                db.update_run_report_path(latest['id'], report_path)
        except Exception as re:
            logger.warning(f"Report generation failed (non-fatal): {re}")

    except Exception as e:
        logger.error(f"MQ backtest error: {e}")
        task_status[task_id]['status'] = 'failed'
        task_status[task_id]['message'] = str(e)


@app.route('/api/agent/screen', methods=['POST'])
def api_agent_run_screening():
    """Start MQ screening in the background."""
    try:
        task_id = f"mq_screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        task_status[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Initializing universe screening...',
            'result': None,
        }

        scheduler.add_job(
            _execute_mq_screening,
            args=[task_id],
            id=task_id,
        )

        return jsonify({'task_id': task_id, 'status': 'started'})
    except Exception as e:
        logger.error(f"MQ screening start error: {e}")
        return jsonify({'error': str(e)}), 500


def _execute_mq_screening(task_id: str):
    """Run MQ screening in background thread."""
    try:
        from services.mq_screening_agent import ScreeningAgent
        from services.mq_agent_db import get_agent_db

        task_status[task_id]['message'] = 'Running momentum + quality screening...'
        task_status[task_id]['progress'] = 10

        agent = ScreeningAgent()
        report = agent.run()

        # Extract regime details and persist to DB
        regime = report.regime
        db = get_agent_db()
        if regime:
            db.save_regime(
                regime=regime.regime,
                index_close=regime.index_close,
                index_200dma=regime.index_200dma,
                vix=regime.vix,
                above_200dma=regime.above_200dma,
                vix_ok=regime.vix_ok,
            )

        task_status[task_id]['status'] = 'completed'
        task_status[task_id]['progress'] = 100
        task_status[task_id]['message'] = (
            f"Scanned {report.total_scanned}: "
            f"{report.momentum_passed} momentum, "
            f"{report.quality_passed} quality"
        )
        screening_result = {
            'total_scanned': report.total_scanned,
            'momentum_passed': report.momentum_passed,
            'quality_passed': report.quality_passed,
            'top_ranked': report.top_ranked[:10],
            'regime': regime.regime if regime else 'UNKNOWN',
            'index_close': regime.index_close if regime else 0,
            'index_200dma': regime.index_200dma if regime else 0,
            'vix': regime.vix if regime else None,
            'run_date': datetime.now().isoformat(),
        }
        task_status[task_id]['result'] = screening_result

        # Persist screening results to file so they survive restarts
        try:
            import json
            screening_file = Path('backtest_data') / 'last_screening.json'
            screening_file.parent.mkdir(parents=True, exist_ok=True)
            screening_file.write_text(json.dumps(screening_result, default=str), encoding='utf-8')
        except Exception as fe:
            logger.warning(f"Could not persist screening results: {fe}")

        # Generate HTML report and save path to DB
        try:
            from services.mq_reporting_agent import ReportingAgent
            reporter = ReportingAgent()
            report_path = reporter.generate_monthly_screening(report)
            # Update the DB run with the report path
            db = get_agent_db()
            latest = db.get_latest_run('screening')
            if latest and report_path:
                db.update_run_report_path(latest['id'], report_path)
        except Exception as re:
            logger.warning(f"Screening report generation failed (non-fatal): {re}")

    except Exception as e:
        logger.error(f"MQ screening error: {e}")
        task_status[task_id]['status'] = 'failed'
        task_status[task_id]['message'] = str(e)


@app.route('/api/agent/status/<task_id>')
def api_agent_task_status(task_id: str):
    """Get status of a running MQ agent task."""
    if task_id not in task_status:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task_status[task_id])


@app.route('/api/agent/signal/<int:signal_id>/dismiss', methods=['POST'])
def api_agent_dismiss_signal(signal_id: int):
    """Dismiss an active signal."""
    try:
        from services.mq_agent_db import get_agent_db
        db = get_agent_db()
        db.dismiss_signal(signal_id)
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"Dismiss signal error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent/report/<int:run_id>')
def api_agent_view_report(run_id: int):
    """Serve the HTML report for a given agent run."""
    try:
        from services.mq_agent_db import get_agent_db
        db = get_agent_db()
        run = db.get_run(run_id)

        if not run or not run.get('report_path'):
            return jsonify({'error': 'Report not found'}), 404

        report_path = Path(run['report_path'])
        if not report_path.exists():
            return jsonify({'error': 'Report file missing'}), 404

        return Response(
            report_path.read_text(encoding='utf-8'),
            mimetype='text/html',
        )
    except Exception as e:
        logger.error(f"Report view error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# MQ Scheduled Jobs
# =============================================================================

def _scheduled_monitoring():
    """Weekday 4:30 PM - run monitoring agent."""
    try:
        from services.mq_monitoring_agent import MonitoringAgent
        from services.mq_reporting_agent import ReportingAgent
        logger.info("[Scheduler] Running daily monitoring...")
        agent = MonitoringAgent()
        report = agent.run(symbols=[])  # Uses DB portfolio positions
        ReportingAgent().generate_daily_brief(report)
        logger.info(f"[Scheduler] Monitoring complete: {len(report.signals)} signals")
    except Exception as e:
        logger.error(f"[Scheduler] Monitoring failed: {e}")


def _scheduled_screening():
    """1st of month 5 PM - run full screening."""
    try:
        from services.mq_screening_agent import ScreeningAgent
        from services.mq_reporting_agent import ReportingAgent
        logger.info("[Scheduler] Running monthly screening...")
        agent = ScreeningAgent()
        report = agent.run()
        ReportingAgent().generate_monthly_screening(report)
        logger.info(f"[Scheduler] Screening complete: {report.quality_passed} quality stocks")
    except Exception as e:
        logger.error(f"[Scheduler] Screening failed: {e}")


def _scheduled_weekly_digest():
    """Sunday 10 AM - weekly digest report."""
    try:
        from services.mq_reporting_agent import ReportingAgent
        from services.mq_agent_db import get_agent_db
        logger.info("[Scheduler] Generating weekly digest...")
        db = get_agent_db()
        runs = db.get_recent_runs(limit=50)
        signals = db.get_active_signals(limit=50)
        ReportingAgent().generate_weekly_digest(runs, signals)
        logger.info("[Scheduler] Weekly digest generated")
    except Exception as e:
        logger.error(f"[Scheduler] Weekly digest failed: {e}")


def _scheduled_rebalance():
    """Jan/Jul 1st 9 AM - semi-annual rebalance."""
    try:
        from services.mq_rebalance_agent import RebalanceAgent
        from services.mq_reporting_agent import ReportingAgent
        logger.info("[Scheduler] Running semi-annual rebalance...")
        agent = RebalanceAgent()
        report = agent.run()
        ReportingAgent().generate_rebalance_report(report)
        logger.info(f"[Scheduler] Rebalance complete: {len(report.exits)} exits, {len(report.entries)} entries")
    except Exception as e:
        logger.error(f"[Scheduler] Rebalance failed: {e}")


# Register scheduled jobs
try:
    scheduler.add_job(
        _scheduled_monitoring,
        'cron', day_of_week='mon-fri', hour=16, minute=30,
        id='mq_monitoring', replace_existing=True,
    )
    scheduler.add_job(
        _scheduled_screening,
        'cron', day='1', hour=17, minute=0,
        id='mq_screening', replace_existing=True,
    )
    scheduler.add_job(
        _scheduled_weekly_digest,
        'cron', day_of_week='sun', hour=10, minute=0,
        id='mq_weekly_digest', replace_existing=True,
    )
    scheduler.add_job(
        _scheduled_rebalance,
        'cron', month='1,7', day='1', hour=9, minute=0,
        id='mq_rebalance', replace_existing=True,
    )
    logger.info("MQ Agent scheduled jobs registered: monitoring, screening, weekly_digest, rebalance")
except Exception as e:
    logger.warning(f"Could not register MQ scheduled jobs: {e}")


# =============================================================================
# KC6 Mean Reversion Routes
# =============================================================================

@app.route('/kc6')
@login_required
def kc6_dashboard():
    """KC6 Mean Reversion Trading Dashboard"""
    return render_template(
        'kc6_dashboard.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/api/kc6/state')
def api_kc6_state():
    """Get KC6 dashboard state: positions, signals, crash filter, stats."""
    try:
        from services.kc6_db import get_kc6_db
        db = get_kc6_db()

        positions = db.get_active_positions()
        daily_state = db.get_daily_state()
        stats = db.get_stats()
        recent_trades = db.get_trade_history(limit=20)

        # Get last scan results from task_status
        last_scan = {}
        for tid, ts in list(task_status.items()):
            if tid.startswith('kc6_scan_') and ts.get('status') == 'completed':
                last_scan = ts.get('result', {})
                break

        return jsonify({
            'positions': positions,
            'daily_state': daily_state,
            'stats': stats,
            'recent_trades': recent_trades,
            'last_scan': last_scan,
            'config': {
                'paper_trading_mode': KC6_DEFAULTS.get('paper_trading_mode', True),
                'live_trading_enabled': KC6_DEFAULTS.get('live_trading_enabled', False),
                'enabled': KC6_DEFAULTS.get('enabled', True),
                'max_positions': KC6_DEFAULTS.get('max_positions', 5),
                'sl_pct': KC6_DEFAULTS.get('sl_pct', 5.0),
                'tp_pct': KC6_DEFAULTS.get('tp_pct', 15.0),
                'max_hold_days': KC6_DEFAULTS.get('max_hold_days', 15),
                'atr_ratio_threshold': KC6_DEFAULTS.get('atr_ratio_threshold', 1.3),
            },
        })
    except Exception as e:
        logger.error(f"KC6 state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kc6/scan', methods=['POST'])
@login_required
def api_kc6_manual_scan():
    """Trigger a manual KC6 scan."""
    try:
        task_id = f"kc6_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_status[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Starting KC6 scan...',
            'result': None,
        }

        scheduler.add_job(
            _execute_kc6_scan,
            args=[task_id],
            id=task_id,
        )

        return jsonify({'task_id': task_id, 'status': 'started'})
    except Exception as e:
        logger.error(f"KC6 scan start error: {e}")
        return jsonify({'error': str(e)}), 500


def _execute_kc6_scan(task_id: str):
    """Run KC6 scan in background."""
    try:
        from services.kc6_scanner import run_full_scan

        task_status[task_id]['message'] = 'Loading data and computing indicators...'
        task_status[task_id]['progress'] = 10

        kite = None
        if is_authenticated():
            kite = get_kite()

        scan = run_full_scan(kite=kite, config=KC6_DEFAULTS)

        task_status[task_id]['status'] = 'completed'
        task_status[task_id]['progress'] = 100
        task_status[task_id]['message'] = (
            f"Scan complete: {scan.get('symbols_loaded', 0)} symbols | "
            f"ATR ratio: {scan.get('universe_atr_ratio', 'N/A')} | "
            f"Entries: {len(scan.get('entries', []))} | "
            f"Exits: {len(scan.get('exits', []))}"
        )
        task_status[task_id]['result'] = scan

    except Exception as e:
        logger.error(f"KC6 scan error: {e}")
        task_status[task_id]['status'] = 'failed'
        task_status[task_id]['message'] = str(e)


@app.route('/api/kc6/scan/status/<task_id>')
@login_required
def api_kc6_scan_status(task_id: str):
    """Get KC6 scan task status."""
    if task_id not in task_status:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task_status[task_id])


@app.route('/api/kc6/kill-switch', methods=['POST'])
@login_required
def api_kc6_kill_switch():
    """Emergency exit ALL KC6 positions."""
    try:
        from services.kc6_executor import KC6Executor
        executor = KC6Executor(config=KC6_DEFAULTS)
        results = executor.emergency_exit_all()
        logger.warning(f"KC6 KILL SWITCH activated: {len(results)} positions closed")
        return jsonify({'status': 'executed', 'results': results})
    except Exception as e:
        logger.error(f"KC6 kill switch error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kc6/trades')
@login_required
def api_kc6_trades():
    """Get KC6 trade history."""
    try:
        from services.kc6_db import get_kc6_db
        db = get_kc6_db()
        limit = request.args.get('limit', 50, type=int)
        trades = db.get_trade_history(limit=limit)
        return jsonify(trades)
    except Exception as e:
        logger.error(f"KC6 trades error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kc6/orders')
@login_required
def api_kc6_orders():
    """Get KC6 order audit log."""
    try:
        from services.kc6_db import get_kc6_db
        db = get_kc6_db()
        limit = request.args.get('limit', 50, type=int)
        orders = db.get_recent_orders(limit=limit)
        return jsonify(orders)
    except Exception as e:
        logger.error(f"KC6 orders error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kc6/equity-curve')
@login_required
def api_kc6_equity_curve():
    """Get equity curve data for chart."""
    try:
        from services.kc6_db import get_kc6_db
        db = get_kc6_db()
        curve = db.get_equity_curve()
        return jsonify(curve)
    except Exception as e:
        logger.error(f"KC6 equity curve error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kc6/toggle-mode', methods=['POST'])
@login_required
def api_kc6_toggle_mode():
    """Toggle between paper and live trading mode."""
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'paper')

        if mode == 'live':
            KC6_DEFAULTS['paper_trading_mode'] = False
            KC6_DEFAULTS['live_trading_enabled'] = True
            logger.warning("KC6: Switched to LIVE trading mode")
        else:
            KC6_DEFAULTS['paper_trading_mode'] = True
            KC6_DEFAULTS['live_trading_enabled'] = False
            logger.info("KC6: Switched to PAPER trading mode")

        return jsonify({
            'mode': mode,
            'paper_trading_mode': KC6_DEFAULTS['paper_trading_mode'],
            'live_trading_enabled': KC6_DEFAULTS['live_trading_enabled'],
        })
    except Exception as e:
        logger.error(f"KC6 toggle mode error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kc6/toggle-enabled', methods=['POST'])
def api_kc6_toggle_enabled():
    """Enable/disable KC6 system."""
    try:
        current = KC6_DEFAULTS.get('enabled', True)
        KC6_DEFAULTS['enabled'] = not current
        status = 'ENABLED' if KC6_DEFAULTS['enabled'] else 'DISABLED'
        logger.info(f"KC6 system {status}")
        return jsonify({'enabled': KC6_DEFAULTS['enabled'], 'status': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# KC6 Scheduled Jobs
# =============================================================================

def _kc6_check_exits():
    """3:15 PM Mon-Fri: Check exit conditions for active positions."""
    try:
        from services.kc6_executor import run_exit_check
        logger.info("[KC6 Scheduler] Running exit check...")
        results = run_exit_check(config=KC6_DEFAULTS)
        logger.info(f"[KC6 Scheduler] Exit check complete: {len(results)} exits")
    except Exception as e:
        logger.error(f"[KC6 Scheduler] Exit check failed: {e}")


def _kc6_full_scan():
    """3:20 PM Mon-Fri: Full scan + execute entries."""
    try:
        from services.kc6_executor import run_entry_scan
        logger.info("[KC6 Scheduler] Running full scan...")
        results = run_entry_scan(config=KC6_DEFAULTS)
        logger.info(f"[KC6 Scheduler] Full scan complete: {len(results)} entries")
    except Exception as e:
        logger.error(f"[KC6 Scheduler] Full scan failed: {e}")


def _kc6_verify_orders():
    """3:25 PM Mon-Fri: Verify order fills/rejections."""
    try:
        from services.kc6_executor import KC6Executor
        logger.info("[KC6 Scheduler] Verifying orders...")
        executor = KC6Executor(config=KC6_DEFAULTS)
        results = executor.verify_orders()
        logger.info(f"[KC6 Scheduler] Order verification: {len(results)} updated")
    except Exception as e:
        logger.error(f"[KC6 Scheduler] Order verification failed: {e}")


def _kc6_position_sync():
    """9:20 AM Mon-Fri: Sync DB positions with Kite holdings."""
    try:
        from services.kc6_executor import KC6Executor
        logger.info("[KC6 Scheduler] Syncing positions...")
        executor = KC6Executor(config=KC6_DEFAULTS)
        result = executor.sync_positions_with_kite()
        logger.info(f"[KC6 Scheduler] Position sync: {result.get('status')}")
    except Exception as e:
        logger.error(f"[KC6 Scheduler] Position sync failed: {e}")


def _kc6_place_targets():
    """9:25 AM Mon-Fri: Place SELL LIMIT orders at KC6 mid for active positions."""
    try:
        from services.kc6_executor import run_place_targets
        logger.info("[KC6 Scheduler] Placing target limit orders...")
        results = run_place_targets(config=KC6_DEFAULTS)
        logger.info(f"[KC6 Scheduler] Target orders placed: {len(results)}")
    except Exception as e:
        logger.error(f"[KC6 Scheduler] Target order placement failed: {e}")


def _kc6_midday_fill_check():
    """12:30 PM Mon-Fri: Check if any target orders filled during the morning."""
    try:
        from services.kc6_executor import KC6Executor
        logger.info("[KC6 Scheduler] Midday target fill check...")
        executor = KC6Executor(config=KC6_DEFAULTS)
        fills = executor.check_target_fills()
        if fills:
            logger.info(f"[KC6 Scheduler] Midday fills: {len(fills)} targets hit")
    except Exception as e:
        logger.error(f"[KC6 Scheduler] Midday fill check failed: {e}")


# Register KC6 scheduled jobs
try:
    scheduler.add_job(
        _kc6_position_sync,
        'cron', day_of_week='mon-fri', hour=9, minute=20,
        id='kc6_position_sync', replace_existing=True,
    )
    scheduler.add_job(
        _kc6_place_targets,
        'cron', day_of_week='mon-fri', hour=9, minute=25,
        id='kc6_place_targets', replace_existing=True,
    )
    scheduler.add_job(
        _kc6_midday_fill_check,
        'cron', day_of_week='mon-fri', hour=12, minute=30,
        id='kc6_midday_fills', replace_existing=True,
    )
    scheduler.add_job(
        _kc6_check_exits,
        'cron', day_of_week='mon-fri', hour=15, minute=15,
        id='kc6_exit_check', replace_existing=True,
    )
    scheduler.add_job(
        _kc6_full_scan,
        'cron', day_of_week='mon-fri', hour=15, minute=20,
        id='kc6_full_scan', replace_existing=True,
    )
    scheduler.add_job(
        _kc6_verify_orders,
        'cron', day_of_week='mon-fri', hour=15, minute=25,
        id='kc6_verify_orders', replace_existing=True,
    )
    logger.info(
        "KC6 scheduled jobs registered: "
        "sync(9:20), targets(9:25), midday(12:30), "
        "exit_check(3:15), full_scan(3:20), verify(3:25)"
    )
except Exception as e:
    logger.warning(f"Could not register KC6 scheduled jobs: {e}")


# =============================================================================
# Collar Paper-Trading (KC6 signal + 3-leg options overlay)
# =============================================================================

@app.route('/collar')
@login_required
def collar_dashboard():
    """Collar (KC6 + options) paper-trading dashboard."""
    return render_template(
        'collar_dashboard.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/api/collar/state')
def api_collar_state():
    """Dashboard state: open collars (with legs + live MTM), signals, stats."""
    try:
        from services.collar_db import get_collar_db
        from services.collar_engine import CollarEngine
        from config import COLLAR_DEFAULTS

        db = get_collar_db()
        engine = CollarEngine(config=COLLAR_DEFAULTS)

        positions = db.get_open_positions_with_legs()
        # Attach MTM snapshot to each position
        for pos in positions:
            try:
                pos['mtm'] = engine.mark_to_market(pos)
            except Exception as e:
                logger.warning(f"[COLLAR] MTM failed for {pos.get('symbol')}: {e}")
                pos['mtm'] = {}

        daily_state = db.get_daily_state()
        stats = db.get_stats()
        recent_trades = db.get_trade_history(limit=20)

        last_scan = {}
        for tid, ts in list(task_status.items()):
            if tid.startswith('collar_scan_') and ts.get('status') == 'completed':
                last_scan = ts.get('result', {}) or {}
                break

        return jsonify({
            'positions': positions,
            'daily_state': daily_state,
            'stats': stats,
            'recent_trades': recent_trades,
            'last_scan': last_scan,
            'config': {
                'paper_trading_mode': COLLAR_DEFAULTS.get('paper_trading_mode', True),
                'enabled': COLLAR_DEFAULTS.get('enabled', True),
                'max_positions': COLLAR_DEFAULTS.get('max_positions', 5),
                'put_otm_pct': COLLAR_DEFAULTS.get('put_otm_pct', 5.0),
                'call_otm_pct': COLLAR_DEFAULTS.get('call_otm_pct', 5.0),
                'sl_pct': COLLAR_DEFAULTS.get('sl_pct', 5.0),
                'tp_pct': COLLAR_DEFAULTS.get('tp_pct', 15.0),
                'max_hold_days': COLLAR_DEFAULTS.get('max_hold_days', 15),
                'iv_assumed': COLLAR_DEFAULTS.get('iv_assumed', 0.25),
            },
        })
    except Exception as e:
        logger.error(f"Collar state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/collar/scan', methods=['POST'])
@login_required
def api_collar_manual_scan():
    """Kick off a manual collar scan (exits + entries, paper-mode)."""
    try:
        task_id = f"collar_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_status[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Starting collar scan...',
            'result': None,
        }
        scheduler.add_job(
            _execute_collar_scan,
            args=[task_id],
            id=task_id,
        )
        return jsonify({'task_id': task_id, 'status': 'started'})
    except Exception as e:
        logger.error(f"Collar scan start error: {e}")
        return jsonify({'error': str(e)}), 500


def _execute_collar_scan(task_id: str):
    try:
        from services.collar_engine import CollarEngine
        from config import COLLAR_DEFAULTS

        task_status[task_id]['message'] = 'Loading data and computing indicators...'
        task_status[task_id]['progress'] = 10

        engine = CollarEngine(config=COLLAR_DEFAULTS)
        result = engine.run_full_scan()

        task_status[task_id]['status'] = 'completed'
        task_status[task_id]['progress'] = 100
        task_status[task_id]['message'] = (
            f"Collar scan complete: {result.get('symbols_loaded', 0)} symbols | "
            f"ATR={result.get('universe_atr_ratio', 'N/A')} | "
            f"Entries={len(result.get('entries_taken', []))} | "
            f"Exits={len(result.get('exits_taken', []))}"
        )
        task_status[task_id]['result'] = result
    except Exception as e:
        logger.error(f"Collar scan error: {e}")
        task_status[task_id]['status'] = 'failed'
        task_status[task_id]['message'] = str(e)


@app.route('/api/collar/scan/status/<task_id>')
@login_required
def api_collar_scan_status(task_id: str):
    if task_id not in task_status:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task_status[task_id])


@app.route('/api/collar/trades')
@login_required
def api_collar_trades():
    try:
        from services.collar_db import get_collar_db
        db = get_collar_db()
        limit = request.args.get('limit', 50, type=int)
        return jsonify(db.get_trade_history(limit=limit))
    except Exception as e:
        logger.error(f"Collar trades error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/collar/equity-curve')
@login_required
def api_collar_equity_curve():
    try:
        from services.collar_db import get_collar_db
        db = get_collar_db()
        return jsonify(db.get_equity_curve())
    except Exception as e:
        logger.error(f"Collar equity curve error: {e}")
        return jsonify({'error': str(e)}), 500


# ---- Scheduler job ----------------------------------------------------------

def _collar_full_scan():
    """3:25 PM Mon-Fri: full collar scan (exits + entries, paper-mode)."""
    try:
        from services.collar_engine import CollarEngine
        from config import COLLAR_DEFAULTS
        if not COLLAR_DEFAULTS.get('enabled', True):
            logger.info("[Collar Scheduler] Disabled — skipping scheduled scan")
            return
        logger.info("[Collar Scheduler] Running full scan...")
        engine = CollarEngine(config=COLLAR_DEFAULTS)
        result = engine.run_full_scan()
        logger.info(
            f"[Collar Scheduler] Scan done: "
            f"entries={len(result.get('entries_taken', []))} "
            f"exits={len(result.get('exits_taken', []))}"
        )
    except Exception as e:
        logger.error(f"[Collar Scheduler] Full scan failed: {e}")


try:
    scheduler.add_job(
        _collar_full_scan,
        'cron', day_of_week='mon-fri', hour=15, minute=25,
        id='collar_full_scan', replace_existing=True,
    )
    logger.info("Collar scheduled job registered: full_scan(15:25 Mon-Fri)")
except Exception as e:
    logger.warning(f"Could not register Collar scheduled job: {e}")


# =============================================================================
# Nifty ORB Strangle (Phase 3 — 8 paper variants)
# =============================================================================

@app.route('/strangle')
def strangle_dashboard():
    """Legacy Jinja page retired 2026-04-26 — redirect to React SPA at /app/strangle."""
    return redirect('/app/strangle', code=302)


def _strangle_today_status(daily_state, has_open):
    """Compute a one-word status badge for today, given a variant's daily-state row."""
    if has_open:
        return 'Open'
    if not daily_state:
        return 'Idle'
    if daily_state.get('exit_taken'):
        return 'Closed'
    if daily_state.get('day_filter_passed') == 0:
        return 'Skip'
    if daily_state.get('signal_seen') and not daily_state.get('rsi_confirmed'):
        return 'Skip'
    if daily_state.get('or_high') is not None:
        return 'Watching'
    return 'Idle'


@app.route('/api/strangle/state')
def api_strangle_state():
    """Top-level state for all 8 variants (compact view for tab strip)."""
    try:
        from services.nifty_strangle_engine import get_strangle_engine
        from services.nifty_strangle_db import get_strangle_db
        from config import STRANGLE_VARIANTS
        from datetime import date as _date

        engine = get_strangle_engine()
        db = get_strangle_db()
        today_str = _date.today().isoformat()

        spot = engine._spot_ltp()
        out = []
        for v in STRANGLE_VARIANTS:
            ds = db.get_daily_state(v['id'], today_str) or {}
            opens = db.get_open_positions(v['id'])
            status = _strangle_today_status(ds, bool(opens))
            out.append({
                'id': v['id'],
                'name': v['name'],
                'or_min': v['or_min'],
                'enabled': v.get('enabled', True),
                'today_status': status,
                'today_pnl': db.get_today_pnl(v['id']),
                'open_positions': len(opens),
            })
        return jsonify({
            'today': today_str,
            'spot_ltp': spot,
            'variants': out,
        })
    except Exception as e:
        logger.error(f"Strangle state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/strangle/variant/<variant_id>')
def api_strangle_variant_detail(variant_id):
    """Detailed state for one variant (for the active tab pane)."""
    try:
        from services.nifty_strangle_engine import get_strangle_engine
        from services.nifty_strangle_db import get_strangle_db
        from config import STRANGLE_VARIANTS_BY_ID
        from datetime import date as _date

        v = STRANGLE_VARIANTS_BY_ID.get(variant_id)
        if not v:
            return jsonify({'error': f'unknown variant: {variant_id}'}), 404

        engine = get_strangle_engine()
        db = get_strangle_db()
        today_str = _date.today().isoformat()

        opens = db.get_open_positions(variant_id)
        open_pos = opens[0] if opens else None
        mtm = engine.snapshot_position_mtm(open_pos) if open_pos else None

        ds = db.get_daily_state(variant_id, today_str) or {}
        stats = db.get_stats(variant_id)
        recent = db.get_trades(variant_id, limit=20)
        today_pnl = db.get_today_pnl(variant_id)

        status = _strangle_today_status(ds, bool(open_pos))

        return jsonify({
            'variant': v,
            'today_status': status,
            'today_pnl': today_pnl,
            'stats': stats,
            'recent_trades': recent,
            'open_position': open_pos,
            'mtm': mtm,
            'daily_state': ds,
        })
    except Exception as e:
        logger.error(f"Strangle variant detail error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/strangle/scan/<variant_id>', methods=['POST'])
def api_strangle_manual_scan(variant_id):
    """Manual entry-scan trigger for one variant."""
    try:
        from services.nifty_strangle_engine import get_strangle_engine
        engine = get_strangle_engine()
        # Run synchronously (entry scan is fast — just one variant)
        result = engine.run_entry_scan(variant_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Strangle manual scan error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/strangle/close/<variant_id>', methods=['POST'])
def api_strangle_close(variant_id):
    """Force-close any open position for a variant (kill switch)."""
    try:
        from services.nifty_strangle_engine import get_strangle_engine
        engine = get_strangle_engine()
        result = engine.run_eod_squareoff(variant_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Strangle close error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/strangle/equity-curve/<variant_id>')
def api_strangle_equity_curve(variant_id):
    """Per-variant cumulative P/L curve."""
    try:
        from services.nifty_strangle_db import get_strangle_db
        db = get_strangle_db()
        return jsonify(db.get_equity_curve(variant_id))
    except Exception as e:
        logger.error(f"Strangle equity curve error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/strangle/stream')
def api_strangle_stream():
    """SSE stream of live spot + per-variant leg LTPs across all open Strangle
    positions. Mirrors the /api/nas/stream pattern but uses kite.ltp() REST
    polling (no WebSocket subscription) since position count is small (≤10).

    Payload shape:
        {type: 'tick', spot: 24042.5, ts: 1714201234.5,
         variants: {
           'or5-std': {pe_now, ce_now, pe_mtm, ce_mtm, net_mtm,
                       pe_tsym, ce_tsym, pe_strike, ce_strike, qty},
           ...
         }}
        {type: 'offline'}  — emitted when Kite isn't authenticated
    """
    import json, time as _time

    def generate():
        from config import STRANGLE_VARIANTS
        from services.nifty_strangle_db import get_strangle_db
        from services.kite_service import get_kite, is_authenticated

        last_snapshot: dict = {}
        yield ": connected\n\n"

        while True:
            try:
                if not is_authenticated():
                    yield f"data: {json.dumps({'type': 'offline'})}\n\n"
                    _time.sleep(5)
                    continue

                db = get_strangle_db()

                # Collect open positions across all variants and the leg
                # tradingsymbols we need quotes for.
                opens = []         # list of dicts (variant_id, position_dict)
                tsyms = set()
                for v in STRANGLE_VARIANTS:
                    for pos in (db.get_open_positions(v['id']) or []):
                        # leg rows in pos['legs'] carry the tradingsymbols
                        pe_tsym = ce_tsym = None
                        for l in (pos.get('legs') or []):
                            if l.get('leg_type') == 'PE':
                                pe_tsym = l.get('tradingsymbol')
                            elif l.get('leg_type') == 'CE':
                                ce_tsym = l.get('tradingsymbol')
                        opens.append({
                            'variant_id': v['id'],
                            'pos': pos,
                            'pe_tsym': pe_tsym,
                            'ce_tsym': ce_tsym,
                        })
                        if pe_tsym:
                            tsyms.add(pe_tsym)
                        if ce_tsym:
                            tsyms.add(ce_tsym)

                # Pull spot + leg LTPs from Kite REST in one round-trip.
                kite = get_kite()
                quote_keys = ['NSE:NIFTY 50'] + [f'NFO:{s}' for s in tsyms]
                ltp_map: dict = {}
                spot = None
                if quote_keys:
                    try:
                        q = kite.ltp(quote_keys) or {}
                        nifty = q.get('NSE:NIFTY 50')
                        if nifty:
                            spot = nifty.get('last_price')
                        for s in tsyms:
                            v = q.get(f'NFO:{s}')
                            if v and v.get('last_price'):
                                ltp_map[s] = float(v['last_price'])
                    except Exception as e:
                        logger.debug(f"[Strangle stream] kite.ltp failed: {e}")

                # Build per-variant snapshot with computed MTM (pe and ce).
                variants_snap: dict = {}
                for o in opens:
                    pos = o['pos']
                    qty = int(pos.get('qty') or pos.get('lot_size') or 0)
                    pe_entry = float(pos.get('pe_entry_price') or 0)
                    ce_entry = float(pos.get('ce_entry_price') or 0)
                    pe_now = ltp_map.get(o['pe_tsym'])
                    ce_now = ltp_map.get(o['ce_tsym'])
                    pe_mtm = ce_mtm = None
                    if pe_now is not None and qty:
                        pe_mtm = round((pe_entry - pe_now) * qty, 2)
                    if ce_now is not None and qty:
                        ce_mtm = round((ce_entry - ce_now) * qty, 2)
                    net = None
                    if pe_mtm is not None and ce_mtm is not None:
                        net = round(pe_mtm + ce_mtm, 2)
                    variants_snap[o['variant_id']] = {
                        'pe_now': round(pe_now, 2) if pe_now is not None else None,
                        'ce_now': round(ce_now, 2) if ce_now is not None else None,
                        'pe_mtm': pe_mtm, 'ce_mtm': ce_mtm, 'net_mtm': net,
                        'pe_tsym': o['pe_tsym'], 'ce_tsym': o['ce_tsym'],
                        'pe_strike': pos.get('pe_strike'),
                        'ce_strike': pos.get('ce_strike'),
                        'qty': qty,
                    }

                snapshot = {
                    'spot': round(spot, 2) if spot else None,
                    'variants': variants_snap,
                }

                # Push only when something changed (saves SSE bandwidth)
                if snapshot != last_snapshot:
                    last_snapshot = snapshot
                    payload = {
                        'type': 'tick',
                        'spot': snapshot['spot'],
                        'variants': snapshot['variants'],
                        'ts': _time.time(),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                _time.sleep(2)  # poll Kite every 2s

            except Exception as e:
                logger.warning(f"[Strangle stream] {e}")
                _time.sleep(5)

    return app.response_class(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


# Master tick scheduler — one cron job that handles all 8 variants
def _strangle_master_tick():
    """Runs every 60s during market hours; dispatches to all 8 variants."""
    try:
        from services.nifty_strangle_engine import get_strangle_engine
        from config import STRANGLE_DEFAULTS
        if not STRANGLE_DEFAULTS.get('enabled', True):
            return
        engine = get_strangle_engine()
        engine.run_master_tick()
    except Exception as e:
        logger.warning(f"[Strangle] master tick error: {e}")


def _strangle_eod_squareoff():
    """15:25 Mon-Fri: EOD square-off for all variants (belt-and-suspenders)."""
    try:
        from services.nifty_strangle_engine import get_strangle_engine
        from config import STRANGLE_VARIANTS
        engine = get_strangle_engine()
        for v in STRANGLE_VARIANTS:
            engine.run_eod_squareoff(v['id'])
    except Exception as e:
        logger.warning(f"[Strangle] EOD squareoff error: {e}")


def _strangle_daily_summary():
    """16:00 Mon-Fri: log a one-line summary per variant."""
    try:
        from services.nifty_strangle_db import get_strangle_db
        from config import STRANGLE_VARIANTS
        from datetime import date as _date
        db = get_strangle_db()
        today_str = _date.today().isoformat()
        for v in STRANGLE_VARIANTS:
            ds = db.get_daily_state(v['id'], today_str) or {}
            pnl = db.get_today_pnl(v['id'])
            logger.info(
                f"[Strangle EOD:{v['id']}] entry_taken={ds.get('entry_taken', 0)} "
                f"exit_taken={ds.get('exit_taken', 0)} reason={ds.get('exit_reason')} "
                f"pnl={pnl:+.2f}"
            )
    except Exception as e:
        logger.warning(f"[Strangle] daily summary error: {e}")


try:
    # Master tick: every 60s during 9:15-15:30 IST Mon-Fri
    scheduler.add_job(
        _strangle_master_tick,
        'cron', day_of_week='mon-fri',
        hour='9-15', minute='*', second='5',
        id='strangle_master_tick', replace_existing=True,
    )
    # Belt-and-suspenders EOD square-off
    scheduler.add_job(
        _strangle_eod_squareoff,
        'cron', day_of_week='mon-fri', hour=15, minute=25,
        id='strangle_eod_squareoff', replace_existing=True,
    )
    scheduler.add_job(
        _strangle_daily_summary,
        'cron', day_of_week='mon-fri', hour=16, minute=0,
        id='strangle_daily_summary', replace_existing=True,
    )
    logger.info(
        "Strangle scheduled jobs registered: master_tick(every 60s 9-15 Mon-Fri), "
        "eod_squareoff(15:25), daily_summary(16:00)"
    )
except Exception as e:
    logger.warning(f"Could not register Strangle scheduled jobs: {e}")


# =============================================================================
# Maruthi Always-On Strategy
# =============================================================================

@app.route('/maruthi')
def maruthi_dashboard():
    """Maruthi Always-On Strategy Dashboard."""
    return render_template(
        'maruthi_dashboard.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/maruthi/algo')
def maruthi_algo_flow():
    """Maruthi strategy algo logic flow diagram."""
    return render_template(
        'maruthi_algo_flow.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/api/maruthi/state')
def api_maruthi_state():
    """Get full Maruthi strategy state."""
    try:
        from services.maruthi_executor import get_maruthi_executor
        executor = get_maruthi_executor(MARUTHI_DEFAULTS)
        return jsonify(executor.get_state())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/maruthi/scan', methods=['POST'])
def api_maruthi_scan():
    """Trigger manual candle check."""
    import threading
    task_id = f"maruthi_scan_{datetime.now().strftime('%H%M%S')}"
    task_status[task_id] = {'status': 'running', 'progress': 0, 'message': 'Starting scan...'}

    def _run(tid):
        try:
            from services.maruthi_executor import get_maruthi_executor
            from services.kite_service import get_kite
            import pandas as pd

            task_status[tid] = {'status': 'running', 'progress': 20, 'message': 'Loading 30-min candles...'}

            executor = get_maruthi_executor(MARUTHI_DEFAULTS)
            symbol = MARUTHI_DEFAULTS.get('symbol', 'MARUTI')

            # Load 30-min data — always try Kite first if authenticated
            from services.kite_service import is_authenticated
            df = pd.DataFrame()

            if is_authenticated():
                try:
                    kite = get_kite()
                    token = 2815745  # MARUTI instrument token
                    from datetime import timedelta
                    to_date = datetime.now()
                    from_date = to_date - timedelta(days=60)
                    data = kite.historical_data(
                        instrument_token=token,
                        from_date=from_date, to_date=to_date,
                        interval='30minute'
                    )
                    df = pd.DataFrame(data)
                except Exception as e:
                    logger.warning(f"Kite historical fetch failed: {e}")

            if df.empty:
                # Fallback: load from DB
                from config import MARKET_DATA_DB
                import sqlite3
                conn = sqlite3.connect(str(MARKET_DATA_DB))
                df = pd.read_sql_query(
                    "SELECT date, open, high, low, close, volume FROM market_data_unified "
                    "WHERE symbol = ? AND timeframe = '30minute' ORDER BY date DESC LIMIT 200",
                    conn, params=(symbol,)
                )
                conn.close()
                df = df.sort_values('date').reset_index(drop=True)

            if df.empty:
                task_status[tid] = {'status': 'error', 'message': f'No data for {symbol}'}
                return

            task_status[tid] = {'status': 'running', 'progress': 60, 'message': 'Computing signals...'}
            results = executor.run_candle_check(df)

            msg = '; '.join(results) if results else 'No signals'
            task_status[tid] = {'status': 'completed', 'progress': 100, 'message': msg}
        except Exception as e:
            logger.error(f"Maruthi scan failed: {e}", exc_info=True)
            task_status[tid] = {'status': 'error', 'message': str(e)}

    threading.Thread(target=_run, args=(task_id,), daemon=True).start()
    return jsonify({'task_id': task_id})


@app.route('/api/maruthi/scan/status/<task_id>')
def api_maruthi_scan_status(task_id):
    """Check Maruthi scan task status."""
    status = task_status.get(task_id)
    if status:
        return jsonify(status)
    return jsonify({'error': 'Task not found'}), 404


@app.route('/api/maruthi/recalc-sl', methods=['POST'])
def api_maruthi_recalc_sl():
    """Force recalculate hard SL from fresh Kite data (resets trailing)."""
    try:
        from services.maruthi_executor import get_maruthi_executor
        from services.maruthi_strategy import compute_dual_supertrend, resolve_sl_buffer, compute_hard_sl
        from services.maruthi_db import get_maruthi_db
        from services.kite_service import get_kite
        from services.nifty500_universe import get_instrument_token
        import pandas as pd

        symbol = MARUTHI_DEFAULTS.get('symbol', 'MARUTI')
        kite = get_kite()
        token = get_instrument_token(symbol)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=30)
        data = kite.historical_data(token, from_date, to_date, '30minute')
        df = pd.DataFrame(data)

        df = compute_dual_supertrend(df,
            master_period=MARUTHI_DEFAULTS.get('master_atr_period', 7),
            master_mult=MARUTHI_DEFAULTS.get('master_multiplier', 5.0),
            child_period=MARUTHI_DEFAULTS.get('child_atr_period', 7),
            child_mult=MARUTHI_DEFAULTS.get('child_multiplier', 2.0),
        )

        last = df.iloc[-1]
        master_st = float(last['master_st'])
        master_atr = float(last['master_atr'])
        sl_buffer = resolve_sl_buffer(MARUTHI_DEFAULTS, master_atr)

        db = get_maruthi_db()
        regime = db.get_regime()
        current_regime = regime.get('regime', 'FLAT')

        # Fresh SL — no trailing (prev_hard_sl=0)
        hard_sl = compute_hard_sl(master_st, current_regime, sl_buffer, prev_hard_sl=0)

        # Update DB and executor
        db.update_regime(master_st_value=master_st, hard_sl_price=hard_sl)
        executor = get_maruthi_executor(MARUTHI_DEFAULTS)
        executor._current_master_atr = master_atr

        return jsonify({
            'status': 'success',
            'master_st': round(master_st, 2),
            'master_atr': round(master_atr, 2),
            'sl_buffer': round(sl_buffer, 2),
            'hard_sl': round(hard_sl, 2),
            'regime': current_regime,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/maruthi/manual-entry', methods=['POST'])
def api_maruthi_manual_entry():
    """Place a manual futures entry order via the live executor."""
    try:
        from services.maruthi_executor import get_maruthi_executor
        from services.maruthi_db import get_maruthi_db

        data = request.get_json()
        direction = data.get('direction', 'SELL')
        trigger_price = float(data['trigger_price'])
        limit_price = float(data.get('limit_price', trigger_price - 2))
        signal_type = data.get('signal_type', 'MANUAL_LIVE_ENTRY')

        db = get_maruthi_db()
        regime = db.get_regime()
        hard_sl = regime.get('hard_sl_price', 0)

        executor = get_maruthi_executor(MARUTHI_DEFAULTS)
        pos_id = executor.place_futures_entry(
            direction=direction,
            trigger_price=trigger_price,
            limit_price=limit_price,
            hard_sl=hard_sl,
            signal_type=signal_type,
            regime=regime.get('regime', 'FLAT'),
        )

        if pos_id:
            return jsonify({
                'status': 'success',
                'position_id': pos_id,
                'direction': direction,
                'trigger_price': trigger_price,
                'limit_price': limit_price,
                'hard_sl': hard_sl,
            })
        else:
            return jsonify({'status': 'error', 'message': 'Order blocked by guardrails'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/maruthi/kill-switch', methods=['POST'])
def api_maruthi_kill_switch():
    """Emergency: close all Maruthi positions at market."""
    try:
        from services.maruthi_executor import get_maruthi_executor
        executor = get_maruthi_executor(MARUTHI_DEFAULTS)
        closed = executor.emergency_exit_all()
        return jsonify({'success': True, 'closed': closed})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/maruthi/trades')
def api_maruthi_trades():
    """Get Maruthi trade history."""
    try:
        from services.maruthi_db import get_maruthi_db
        db = get_maruthi_db()
        return jsonify(db.get_recent_trades(50))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/maruthi/orders')
def api_maruthi_orders():
    """Get Maruthi order audit log."""
    try:
        from services.maruthi_db import get_maruthi_db
        db = get_maruthi_db()
        orders = db.get_pending_orders()
        return jsonify(orders)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/maruthi/signals')
def api_maruthi_signals():
    """Get Maruthi signal history."""
    try:
        from services.maruthi_db import get_maruthi_db
        db = get_maruthi_db()
        return jsonify(db.get_recent_signals(50))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/maruthi/equity-curve')
def api_maruthi_equity_curve():
    """Get Maruthi cumulative PnL for charting."""
    try:
        from services.maruthi_db import get_maruthi_db
        db = get_maruthi_db()
        return jsonify(db.get_equity_curve())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/maruthi/mtm')
def api_maruthi_mtm():
    """Get live MTM P&L for all active positions."""
    try:
        from services.maruthi_db import get_maruthi_db
        from services.kite_service import get_kite, is_authenticated

        db = get_maruthi_db()
        all_pos = db.get_active_positions()

        if not all_pos:
            return jsonify({'positions': [], 'total_mtm': 0, 'total_unrealized': 0})

        # Try to get live prices from ticker first, then Kite API
        ltp_map = {}
        try:
            from services.maruthi_ticker import get_maruthi_ticker
            ticker = get_maruthi_ticker(MARUTHI_DEFAULTS)
            if ticker._last_ltp > 0:
                ltp_map['SPOT'] = ticker._last_ltp
        except Exception:
            pass

        # For live mode or if we need option prices, use Kite quotes
        if is_authenticated():
            try:
                kite = get_kite()
                symbols_to_quote = []
                for pos in all_pos:
                    sym = pos['tradingsymbol']
                    if '_PAPER' not in sym:
                        symbols_to_quote.append(f"NFO:{sym}")

                if symbols_to_quote:
                    quotes = kite.quote(symbols_to_quote)
                    for key, q in quotes.items():
                        sym = key.split(':')[1] if ':' in key else key
                        ltp_map[sym] = q.get('last_price', 0)
            except Exception as e:
                logger.warning(f"MTM quote fetch failed: {e}")

        # Compute MTM for each position
        spot_ltp = ltp_map.get('SPOT', 0)
        result_positions = []
        total_unrealized = 0

        for pos in all_pos:
            sym = pos['tradingsymbol']
            entry = pos['entry_price'] or 0
            qty = pos['qty'] or 0

            # Get LTP for this instrument
            if '_PAPER' in sym and pos['position_type'] == 'FUTURES':
                ltp = spot_ltp
            else:
                ltp = ltp_map.get(sym, 0)

            # Calculate unrealized P&L
            if ltp > 0 and entry > 0:
                if pos['transaction_type'] == 'BUY':
                    pnl = (ltp - entry) * qty
                else:  # SELL
                    pnl = (entry - ltp) * qty
            else:
                pnl = 0

            total_unrealized += pnl

            result_positions.append({
                'id': pos['id'],
                'position_type': pos['position_type'],
                'tradingsymbol': sym,
                'transaction_type': pos['transaction_type'],
                'qty': qty,
                'entry_price': entry,
                'ltp': ltp,
                'pnl': round(pnl, 2),
                'pnl_pct': round((pnl / (entry * qty) * 100), 2) if entry * qty > 0 else 0,
                'sl_price': pos.get('sl_price'),
                'strike': pos.get('strike'),
                'expiry_date': pos.get('expiry_date'),
                'status': pos.get('status'),
            })

        # Add realized P&L from closed trades
        stats = db.get_stats()
        realized = stats.get('total_pnl', 0)

        return jsonify({
            'positions': result_positions,
            'total_unrealized': round(total_unrealized, 2),
            'total_realized': round(realized, 2),
            'total_mtm': round(total_unrealized + realized, 2),
            'spot_ltp': spot_ltp,
        })
    except Exception as e:
        logger.error(f"MTM error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/maruthi/toggle-mode', methods=['POST'])
def api_maruthi_toggle_mode():
    """Toggle Maruthi between paper and live trading. Persists to DB."""
    global MARUTHI_DEFAULTS
    if MARUTHI_DEFAULTS.get('paper_trading_mode'):
        MARUTHI_DEFAULTS['paper_trading_mode'] = False
        MARUTHI_DEFAULTS['live_trading_enabled'] = True
        mode = 'LIVE'
    else:
        MARUTHI_DEFAULTS['paper_trading_mode'] = True
        MARUTHI_DEFAULTS['live_trading_enabled'] = False
        mode = 'PAPER'
    # Persist to DB
    try:
        from services.maruthi_db import get_maruthi_db
        db = get_maruthi_db()
        db.set_setting('trading_mode', mode)
    except Exception as e:
        logger.warning(f"Failed to persist Maruthi mode: {e}")
    return jsonify({'mode': mode})


@app.route('/api/maruthi/toggle-enabled', methods=['POST'])
def api_maruthi_toggle_enabled():
    """Enable/disable Maruthi system. Auto-starts/stops ticker. Persists to DB."""
    try:
        # Toggle enabled state (default True if not set)
        current = MARUTHI_DEFAULTS.get('enabled', True)
        MARUTHI_DEFAULTS['enabled'] = not current
        new_enabled = MARUTHI_DEFAULTS['enabled']
        status = 'ENABLED' if new_enabled else 'DISABLED'
        logger.info(f"Maruthi system {status}")
        # Persist to DB
        try:
            from services.maruthi_db import get_maruthi_db
            db = get_maruthi_db()
            db.set_setting('enabled', str(new_enabled))
        except Exception:
            pass

        # Auto-start/stop ticker
        if new_enabled:
            try:
                from services.maruthi_ticker import get_maruthi_ticker
                ticker = get_maruthi_ticker(MARUTHI_DEFAULTS)
                if not ticker.is_connected:
                    ticker.start()
                    logger.info("[Maruthi] Ticker auto-started on enable")
            except Exception as te:
                logger.warning(f"[Maruthi] Ticker auto-start failed: {te}")
        else:
            try:
                from services.maruthi_ticker import get_maruthi_ticker
                ticker = get_maruthi_ticker()
                if ticker.is_connected:
                    ticker.stop()
                    logger.info("[Maruthi] Ticker stopped on disable")
            except Exception as te:
                logger.warning(f"[Maruthi] Ticker stop failed: {te}")

        return jsonify({'enabled': new_enabled, 'status': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/maruthi/auth', methods=['POST'])
def api_maruthi_auth():
    """Trigger TOTP auto-login."""
    try:
        from services.kite_auth import ensure_authenticated
        success = ensure_authenticated()
        return jsonify({'authenticated': success})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- Maruthi WebSocket Ticker + Postback + Scheduled Jobs ----

# Postback endpoint — Kite POSTs here when order status changes
@app.route('/api/maruthi/postback', methods=['POST'])
def api_maruthi_postback():
    """
    Kite order postback webhook.
    Configure this URL in your Kite Connect app settings:
    https://your-domain.com/api/maruthi/postback

    Kite sends JSON like:
    {
        "order_id": "123456",
        "status": "COMPLETE",  // or CANCELLED, REJECTED
        "average_price": 7045.0,
        "filled_quantity": 200,
        "tradingsymbol": "MARUTI25MARFUT",
        ...
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        order_id = data.get('order_id', '')
        status = data.get('status', '')
        avg_price = data.get('average_price', 0)
        tradingsymbol = data.get('tradingsymbol', '')

        logger.info(f"[Maruthi Postback] order={order_id} status={status} "
                     f"symbol={tradingsymbol} price={avg_price}")

        if not order_id:
            return jsonify({'status': 'ignored', 'reason': 'no order_id'}), 200

        from services.maruthi_db import get_maruthi_db
        db = get_maruthi_db()

        if status == 'COMPLETE':
            # Check if this is a pending trigger order that just filled
            pending = db.get_pending_positions()
            for pos in pending:
                if pos.get('kite_order_id') == str(order_id):
                    fill_price = avg_price or pos.get('trigger_price', 0)
                    db.activate_position(pos['id'], fill_price)
                    logger.info(f"[Maruthi Postback] Position {pos['id']} activated: "
                                f"{tradingsymbol} @ {fill_price}")
                    break

            # Also update the order log
            pending_orders = db.get_pending_orders()
            for order in pending_orders:
                if order.get('kite_order_id') == str(order_id):
                    db.update_order(order['id'], status='FILLED', price=avg_price)
                    break

        elif status in ('CANCELLED', 'REJECTED'):
            pending = db.get_pending_positions()
            for pos in pending:
                if pos.get('kite_order_id') == str(order_id):
                    db.cancel_position(pos['id'])
                    logger.warning(f"[Maruthi Postback] Position {pos['id']} cancelled: "
                                   f"{tradingsymbol} ({status})")
                    break

            pending_orders = db.get_pending_orders()
            for order in pending_orders:
                if order.get('kite_order_id') == str(order_id):
                    reason = data.get('status_message', status)
                    db.update_order(order['id'], status=status, error_message=reason)
                    break

        return jsonify({'status': 'ok'}), 200

    except Exception as e:
        logger.error(f"[Maruthi Postback] Error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 200  # Return 200 so Kite doesn't retry


# WebSocket ticker control endpoints
@app.route('/api/maruthi/ticker/start', methods=['POST'])
def api_maruthi_ticker_start():
    """Start the WebSocket ticker for live candle streaming."""
    try:
        from services.maruthi_ticker import get_maruthi_ticker
        ticker = get_maruthi_ticker(MARUTHI_DEFAULTS)
        ticker.start()
        return jsonify({'status': 'started', 'ticker': ticker.get_status()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/maruthi/ticker/stop', methods=['POST'])
def api_maruthi_ticker_stop():
    """Stop the WebSocket ticker."""
    try:
        from services.maruthi_ticker import get_maruthi_ticker
        ticker = get_maruthi_ticker()
        ticker.stop()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/maruthi/ticker/status')
def api_maruthi_ticker_status():
    """Get ticker connection status."""
    try:
        from services.maruthi_ticker import get_maruthi_ticker
        ticker = get_maruthi_ticker(MARUTHI_DEFAULTS)
        return jsonify(ticker.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# ---- Scheduled Jobs (only time-based, no candle polling) ----

def _maruthi_auto_login_and_start():
    """9:00 AM — TOTP auto-login, then start WebSocket ticker."""
    try:
        from services.kite_auth import ensure_authenticated
        if ensure_authenticated():
            logger.info("[Maruthi] TOTP auto-login successful")
            # Start or restart ticker with fresh token
            from services.maruthi_ticker import get_maruthi_ticker
            ticker = get_maruthi_ticker(MARUTHI_DEFAULTS)
            ticker.restart()
            logger.info("[Maruthi] WebSocket ticker started")
        else:
            logger.error("[Maruthi] TOTP auto-login FAILED — ticker not started")
    except Exception as e:
        logger.error(f"[Maruthi] Auto-login error: {e}")


def _maruthi_eod_protection():
    """3:00 PM — Buy protective options for unhedged futures."""
    try:
        from services.maruthi_executor import get_maruthi_executor
        executor = get_maruthi_executor(MARUTHI_DEFAULTS)

        if MARUTHI_DEFAULTS.get('paper_trading_mode'):
            logger.info("[Maruthi] Paper mode — skipping EOD protection")
            return

        # Get spot from ticker LTP
        from services.maruthi_ticker import get_maruthi_ticker
        ticker = get_maruthi_ticker(MARUTHI_DEFAULTS)
        spot = ticker._last_ltp

        if spot <= 0:
            # Fallback: fetch from Kite quote API
            from services.kite_service import get_kite
            kite = get_kite()
            symbol = MARUTHI_DEFAULTS.get('symbol', 'MARUTI')
            quote = kite.quote([f"NSE:{symbol}"])
            spot = quote.get(f"NSE:{symbol}", {}).get('last_price', 0)

        if spot > 0:
            results = executor.run_eod_protection(spot)
            if results:
                logger.info(f"[Maruthi] EOD protection: {results}")
    except Exception as e:
        logger.error(f"[Maruthi] EOD protection failed: {e}")


def _maruthi_roll_check():
    """3:15 PM — Check for contracts nearing expiry and roll."""
    try:
        from services.maruthi_executor import get_maruthi_executor
        executor = get_maruthi_executor(MARUTHI_DEFAULTS)
        results = executor.run_roll_check()
        if results:
            logger.info(f"[Maruthi] Rolls: {results}")
    except Exception as e:
        logger.error(f"[Maruthi] Roll check failed: {e}")


def _maruthi_re_place_pending():
    """9:16 AM — Re-place pending trigger orders on Kite (Zerodha cancels unfilled SL-L at EOD)."""
    try:
        from services.maruthi_executor import get_maruthi_executor
        executor = get_maruthi_executor(MARUTHI_DEFAULTS)
        results = executor.re_place_pending_orders()
        if results:
            logger.info(f"[Maruthi] Re-placed pending orders: {results}")
    except Exception as e:
        logger.error(f"[Maruthi] Re-place pending orders failed: {e}")


def _maruthi_gap_handler():
    """9:21 AM — Handle GAP_PENDING positions after 5 minutes of market confirmation."""
    try:
        from services.maruthi_executor import get_maruthi_executor
        executor = get_maruthi_executor(MARUTHI_DEFAULTS)
        results = executor.handle_gap_entry()
        if results:
            logger.info(f"[Maruthi] Gap handler: {results}")
    except Exception as e:
        logger.error(f"[Maruthi] Gap handler failed: {e}")


def _maruthi_market_close():
    """3:30 PM — Force-close current candle, stop ticker."""
    try:
        from services.maruthi_ticker import get_maruthi_ticker
        ticker = get_maruthi_ticker(MARUTHI_DEFAULTS)
        ticker.aggregator.force_close()
        logger.info("[Maruthi] Forced final candle close at market close")
        # Don't stop ticker — let it disconnect naturally
    except Exception as e:
        logger.error(f"[Maruthi] Market close handler failed: {e}")


# Register Maruthi scheduled jobs — DISABLED until algo bugs are fixed
# To re-enable: set MARUTHI_DEFAULTS['enabled'] = True in config.py
if MARUTHI_DEFAULTS.get('enabled', False):
    try:
        scheduler.add_job(
            _maruthi_auto_login_and_start,
            'cron', day_of_week='mon-fri', hour=9, minute=0,
            id='maruthi_auto_login', replace_existing=True,
        )
        scheduler.add_job(
            _maruthi_re_place_pending,
            'cron', day_of_week='mon-fri', hour=9, minute=16,
            id='maruthi_re_place_pending', replace_existing=True,
        )
        scheduler.add_job(
            _maruthi_gap_handler,
            'cron', day_of_week='mon-fri', hour=9, minute=21,
            id='maruthi_gap_handler', replace_existing=True,
        )
        scheduler.add_job(
            _maruthi_eod_protection,
            'cron', day_of_week='mon-fri', hour=15, minute=0,
            id='maruthi_eod_protection', replace_existing=True,
        )
        scheduler.add_job(
            _maruthi_roll_check,
            'cron', day_of_week='mon-fri', hour=15, minute=15,
            id='maruthi_roll_check', replace_existing=True,
        )
        scheduler.add_job(
            _maruthi_market_close,
            'cron', day_of_week='mon-fri', hour=15, minute=30,
            id='maruthi_market_close', replace_existing=True,
        )
        logger.info(
            "Maruthi scheduled jobs registered: "
            "auto-login+ticker(9:00), re-place pending(9:16), gap handler(9:21), "
            "EOD protection(15:00), roll check(15:15), market close(15:30). "
            "Candle signals handled by WebSocket ticker."
        )
    except Exception as e:
        logger.warning(f"Could not register Maruthi scheduled jobs: {e}")
else:
    logger.info("Maruthi algo DISABLED (config.enabled=False) — no scheduled jobs, no ticker, no live trades")


# =============================================================================
# BNF Squeeze & Fire — Dashboard & API
# =============================================================================

@app.route('/bnf')
def bnf_dashboard():
    """BNF Squeeze & Fire execution dashboard."""
    return render_template(
        'bnf_dashboard.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/api/bnf/state')
def api_bnf_state():
    """Get full BNF system state for dashboard."""
    try:
        from services.bnf_executor import get_bnf_executor
        executor = get_bnf_executor(BNF_DEFAULTS)
        return jsonify(executor.get_full_state())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bnf/scan', methods=['POST'])
def api_bnf_scan():
    """Trigger manual BNF scan (async)."""
    import uuid
    task_id = str(uuid.uuid4())[:8]

    def _run_scan(tid):
        try:
            from services.bnf_executor import get_bnf_executor
            executor = get_bnf_executor(BNF_DEFAULTS)
            result = executor.run_daily_scan()
            _bnf_scan_results[tid] = {'status': 'complete', 'result': result}
        except Exception as e:
            _bnf_scan_results[tid] = {'status': 'error', 'error': str(e)}

    _bnf_scan_results[task_id] = {'status': 'running'}
    scheduler.add_job(_run_scan, args=[task_id], id=f'bnf_scan_{task_id}',
                      replace_existing=True)
    return jsonify({'task_id': task_id, 'status': 'started'})


# Store scan results
_bnf_scan_results = {}


@app.route('/api/bnf/scan/status/<task_id>')
def api_bnf_scan_status(task_id):
    """Check BNF scan task status."""
    result = _bnf_scan_results.get(task_id, {'status': 'unknown'})
    return jsonify(result)


@app.route('/api/bnf/kill-switch', methods=['POST'])
def api_bnf_kill_switch():
    """Emergency exit all BNF positions."""
    try:
        from services.bnf_executor import get_bnf_executor
        executor = get_bnf_executor(BNF_DEFAULTS)
        closed = executor.emergency_exit_all()
        return jsonify({'closed': closed, 'status': 'OK'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bnf/trades')
def api_bnf_trades():
    """Get recent BNF trades."""
    try:
        from services.bnf_db import get_bnf_db
        db = get_bnf_db()
        trades = db.get_recent_trades(limit=50)
        return jsonify(trades)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bnf/orders')
def api_bnf_orders():
    """Get recent BNF orders."""
    try:
        from services.bnf_db import get_bnf_db
        db = get_bnf_db()
        orders = db.get_pending_orders()
        return jsonify(orders)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bnf/signals')
def api_bnf_signals():
    """Get recent BNF signals."""
    try:
        from services.bnf_db import get_bnf_db
        db = get_bnf_db()
        signals = db.get_recent_signals(limit=30)
        return jsonify(signals)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bnf/equity-curve')
def api_bnf_equity_curve():
    """Get BNF equity curve."""
    try:
        from services.bnf_db import get_bnf_db
        db = get_bnf_db()
        curve = db.get_equity_curve()
        return jsonify(curve)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bnf/toggle-mode', methods=['POST'])
def api_bnf_toggle_mode():
    """Toggle BNF paper/live trading mode."""
    try:
        current = BNF_DEFAULTS.get('paper_trading_mode', True)
        BNF_DEFAULTS['paper_trading_mode'] = not current
        new_mode = 'PAPER' if BNF_DEFAULTS['paper_trading_mode'] else 'LIVE'
        logger.info(f"BNF mode toggled to {new_mode}")
        return jsonify({'mode': new_mode})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bnf/toggle-enabled', methods=['POST'])
def api_bnf_toggle_enabled():
    """Enable/disable BNF system."""
    try:
        current = BNF_DEFAULTS.get('enabled', True)
        BNF_DEFAULTS['enabled'] = not current
        status = 'ENABLED' if BNF_DEFAULTS['enabled'] else 'DISABLED'
        logger.info(f"BNF system {status}")
        return jsonify({'enabled': BNF_DEFAULTS['enabled'], 'status': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- BNF Scheduled Jobs ----

def _bnf_daily_scan():
    """3:20 PM Mon-Fri — Run BNF daily scan after market close."""
    if not BNF_DEFAULTS.get('enabled', True):
        logger.info("[BNF] Skipping daily scan — strategy disabled")
        return
    try:
        from services.bnf_executor import get_bnf_executor
        executor = get_bnf_executor(BNF_DEFAULTS)
        result = executor.run_daily_scan()
        entries = len(result.get('entries', []))
        exits = len(result.get('exits', []))
        logger.info(f"[BNF] Daily scan: {entries} entries, {exits} exits")
    except Exception as e:
        logger.error(f"[BNF] Daily scan error: {e}")


def _bnf_exit_check():
    """3:15 PM Mon-Fri — Check BNF positions for exits (before new entries)."""
    if not BNF_DEFAULTS.get('enabled', True):
        return
    try:
        from services.bnf_executor import get_bnf_executor
        executor = get_bnf_executor(BNF_DEFAULTS)
        scan = executor.scanner.scan()
        exits = executor.check_and_exit(scan)
        if exits:
            logger.info(f"[BNF] Exit check: {len(exits)} positions closed")
    except Exception as e:
        logger.error(f"[BNF] Exit check error: {e}")


try:
    scheduler.add_job(
        _bnf_exit_check,
        'cron', day_of_week='mon-fri', hour=15, minute=15,
        id='bnf_exit_check', replace_existing=True,
    )
    scheduler.add_job(
        _bnf_daily_scan,
        'cron', day_of_week='mon-fri', hour=15, minute=20,
        id='bnf_daily_scan', replace_existing=True,
    )
    logger.info(
        "BNF scheduled jobs registered: "
        "exit check(15:15), daily scan(15:20)"
    )
except Exception as e:
    logger.warning(f"Could not register BNF scheduled jobs: {e}")


# =============================================================================
# NAS — Nifty ATR Strangle (Intraday Options Selling)
# =============================================================================

# Background task store for NAS scans
_nas_tasks = {}


@app.route('/nas')
def nas_dashboard():
    """NAS Combined Dashboard (OTM + ATM side-by-side)."""
    # Auto-start NAS ticker when visiting dashboard (if authenticated)
    if is_authenticated():
        try:
            from services.nas_ticker import get_nas_ticker
            ticker = get_nas_ticker(NAS_DEFAULTS)
            if not ticker.is_connected:
                ticker.restart()
                logger.info("[NAS] Ticker auto-started on dashboard visit")
        except Exception as e:
            logger.warning(f"[NAS] Ticker auto-start on visit failed: {e}")
    return render_template(
        'nas_combined_dashboard.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


def _enrich_nas_positions_with_ltp(state, ticker_attr='_option_ltps', token_attr='_option_tokens'):
    """Enrich position dicts with live LTP + pnl_inr + live spot from NAS ticker cache.

    ticker_attr / token_attr select which system's cached LTPs to use.
    """
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker()
        # Live Nifty spot from ticker WebSocket (updates every tick)
        live_spot = getattr(ticker, '_last_ltp', None)
        if live_spot and live_spot > 0:
            st = state.setdefault('state', {})
            st['spot_price'] = round(live_spot, 2)

        # Attach available margin (shared with ORB via _orb_cache, 5-min refresh)
        try:
            state['margin'] = _orb_get_margin() or {}
        except Exception:
            pass

        # Realized P&L from positions closed today — survives after EOD squareoff.
        # Sums (entry - exit) * qty across all closed-today legs (NAS shorts options).
        try:
            closed_today = (state.get('positions') or {}).get('closed_today') or []
            realized = 0.0
            for p in closed_today:
                entry = p.get('entry_price') or p.get('entry_premium') or 0
                exit_price = p.get('exit_price') or 0
                qty = p.get('qty') or 0
                if exit_price and qty:
                    realized += (entry - exit_price) * qty
            stats = state.setdefault('stats', {})
            stats['today_pnl'] = round(realized, 2)
        except Exception:
            pass

        ltps_by_token = getattr(ticker, ticker_attr, {}) or {}
        tokens_by_tsym = {
            info.get('tradingsymbol'): token
            for token, info in getattr(ticker, token_attr, {}).items()
        }
        positions = state.get('positions') or {}
        for leg_key in ('ce', 'pe'):
            for p in positions.get(leg_key, []) or []:
                tsym = p.get('tradingsymbol')
                if not tsym:
                    continue
                token = tokens_by_tsym.get(tsym)
                ltp = ltps_by_token.get(token) if token else None
                if ltp is None or ltp <= 0:
                    continue
                p['ltp'] = round(ltp, 2)
                entry = p.get('entry_price') or p.get('entry_premium')
                qty = p.get('qty') or p.get('lots') or 0
                if entry is not None and qty:
                    # NAS sells options — profit when LTP falls below entry
                    p['pnl_inr'] = round((entry - ltp) * qty, 2)
    except Exception:
        pass
    return state


@app.route('/api/nas/state')
def api_nas_state():
    """Full state dump for NAS dashboard."""
    try:
        from services.nas_executor import NasExecutor
        executor = NasExecutor(config=NAS_DEFAULTS)
        state = executor.get_full_state()
        state = _enrich_nas_positions_with_ltp(state, '_option_ltps', '_option_tokens')
        return jsonify(state)
    except Exception as e:
        logger.error(f"NAS state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/scan', methods=['POST'])
def api_nas_scan():
    """Trigger a manual NAS scan."""
    task_id = f"nas_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _nas_tasks[task_id] = {'status': 'running'}

    def _run_scan(tid):
        try:
            from services.nas_executor import NasExecutor
            executor = NasExecutor(config=NAS_DEFAULTS)
            result = executor.run_scan()
            _nas_tasks[tid] = {'status': 'completed', 'result': result}
        except Exception as e:
            logger.error(f"NAS scan error: {e}")
            _nas_tasks[tid] = {'status': 'error', 'error': str(e)}

    scheduler.add_job(_run_scan, args=[task_id], id=f'nas_scan_{task_id}',
                      replace_existing=True)
    return jsonify({'task_id': task_id, 'status': 'queued'})


@app.route('/api/nas/scan/status/<task_id>')
def api_nas_scan_status(task_id):
    """Poll NAS scan status."""
    import numpy as np
    task = _nas_tasks.get(task_id, {'status': 'unknown'})
    # Convert numpy types to native Python for JSON serialization
    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
    return jsonify(_convert(task))


@app.route('/api/nas/kill-switch', methods=['POST'])
def api_nas_kill_switch():
    """Emergency close all NAS positions."""
    try:
        from services.nas_executor import NasExecutor
        executor = NasExecutor(config=NAS_DEFAULTS)
        closed = executor.emergency_exit_all()
        return jsonify({'closed': closed, 'status': 'EMERGENCY_EXIT'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/trades')
def api_nas_trades():
    """Recent NAS trades."""
    try:
        from services.nas_db import get_nas_db
        db = get_nas_db()
        return jsonify(db.get_recent_trades(limit=50))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/orders')
def api_nas_orders():
    """NAS order audit log."""
    try:
        from services.nas_db import get_nas_db
        db = get_nas_db()
        with db.db_lock:
            conn = db._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM nas_orders ORDER BY created_at DESC LIMIT 50"
                ).fetchall()
                return jsonify([dict(r) for r in rows])
            finally:
                conn.close()
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/signals')
def api_nas_signals():
    """NAS signal history."""
    try:
        from services.nas_db import get_nas_db
        db = get_nas_db()
        return jsonify(db.get_recent_signals(limit=30))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/equity-curve')
def api_nas_equity_curve():
    """NAS daily P&L for equity curve chart."""
    try:
        from services.nas_db import get_nas_db
        db = get_nas_db()
        return jsonify(db.get_equity_curve())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/master-mode', methods=['GET', 'POST'])
def api_nas_master_mode():
    """One-stop master switch for all 8 NAS systems.

    GET returns the consolidated mode across the 8 variants:
      'live'   — every variant has enabled=True AND paper_trading_mode=False
      'paper'  — every variant has enabled=True AND paper_trading_mode=True
      'off'    — every variant has enabled=False
      'mixed'  — variants are in a mix of states (e.g. some live, some paper)

    POST {"mode": "live" | "paper" | "off"} forces all 8 variants to that
    state simultaneously. Returns the resulting consolidated mode.

    Implementation: writes to the in-process *_DEFAULTS dicts directly so
    each variant's _check_guardrails sees the new state on its next entry
    attempt. No restart needed.
    """
    variants = [
        ('NAS Squeeze OTM',    NAS_DEFAULTS),
        ('NAS Squeeze ATM',    NAS_ATM_DEFAULTS),
        ('NAS Squeeze ATM2',   NAS_ATM2_DEFAULTS),
        ('NAS Squeeze ATM4',   NAS_ATM4_DEFAULTS),
        ('NAS 9:16 OTM',       NAS_916_OTM_DEFAULTS),
        ('NAS 9:16 ATM',       NAS_916_ATM_DEFAULTS),
        ('NAS 9:16 ATM2',      NAS_916_ATM2_DEFAULTS),
        ('NAS 9:16 ATM4',      NAS_916_ATM4_DEFAULTS),
    ]

    def consolidate():
        states = []
        for _, cfg in variants:
            en = bool(cfg.get('enabled', True))
            paper = bool(cfg.get('paper_trading_mode', True))
            if not en:
                states.append('off')
            elif paper:
                states.append('paper')
            else:
                states.append('live')
        unique = set(states)
        if len(unique) == 1:
            return next(iter(unique))
        return 'mixed'

    if request.method == 'GET':
        return jsonify({
            'mode': consolidate(),
            'variants': [
                {
                    'name': name,
                    'enabled': bool(cfg.get('enabled', True)),
                    'paper': bool(cfg.get('paper_trading_mode', True)),
                }
                for name, cfg in variants
            ],
        })

    # POST
    try:
        data = request.get_json(silent=True) or {}
        mode = (data.get('mode') or '').lower().strip()
        if mode not in ('off', 'paper', 'live'):
            return jsonify({'error': 'mode must be one of: off, paper, live'}), 400

        for name, cfg in variants:
            if mode == 'off':
                cfg['enabled'] = False
                # leave paper flag as-is so a future flip to "paper" or "live"
                # doesn't accidentally lose paper-only safety
            elif mode == 'paper':
                cfg['enabled'] = True
                cfg['paper_trading_mode'] = True
            elif mode == 'live':
                cfg['enabled'] = True
                cfg['paper_trading_mode'] = False
            logger.info(f"[NAS-MASTER] {name}: enabled={cfg['enabled']} paper={cfg['paper_trading_mode']}")

        result_mode = consolidate()
        logger.warning(f"[NAS-MASTER] All 8 NAS systems set to '{mode}' (consolidated={result_mode})")
        return jsonify({
            'mode': result_mode,
            'requested': mode,
        })
    except Exception as e:
        logger.error(f"[NAS-MASTER] error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/toggle-mode', methods=['POST'])
def api_nas_toggle_mode():
    """Toggle NAS paper/live trading mode."""
    try:
        current = NAS_DEFAULTS.get('paper_trading_mode', True)
        NAS_DEFAULTS['paper_trading_mode'] = not current
        new_mode = 'PAPER' if NAS_DEFAULTS['paper_trading_mode'] else 'LIVE'
        logger.info(f"NAS mode toggled to {new_mode}")
        return jsonify({'mode': new_mode})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/toggle-enabled', methods=['POST'])
def api_nas_toggle_enabled():
    """Enable/disable NAS system. Starts/stops WebSocket ticker accordingly."""
    try:
        current = NAS_DEFAULTS.get('enabled', True)
        NAS_DEFAULTS['enabled'] = not current
        new_enabled = NAS_DEFAULTS['enabled']
        status = 'ENABLED' if new_enabled else 'DISABLED'
        logger.info(f"NAS system {status}")

        # Start/stop ticker with enable/disable
        if new_enabled:
            try:
                from services.nas_ticker import get_nas_ticker
                ticker = get_nas_ticker(NAS_DEFAULTS)
                if not ticker.is_connected:
                    ticker.start()
                    logger.info("[NAS] Ticker auto-started on enable")
            except Exception as te:
                logger.warning(f"[NAS] Ticker auto-start failed: {te}")
        else:
            try:
                from services.nas_ticker import get_nas_ticker
                ticker = get_nas_ticker()
                if ticker.is_connected:
                    ticker.stop()
                    logger.info("[NAS] Ticker stopped on disable")
            except Exception as te:
                logger.warning(f"[NAS] Ticker stop failed: {te}")

        return jsonify({'enabled': new_enabled, 'status': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/config', methods=['POST'])
def api_nas_config_update():
    """Hot-update NAS config values without restarting Flask."""
    data = request.get_json() or {}
    allowed = {'max_vix', 'min_combined_premium', 'strike_distance_atr',
               'premium_double_trigger', 'premium_half_trigger', 'max_daily_loss',
               'lots_per_leg', 'max_adjustments_per_leg',
               'profit_target_pct', 'entry_start_time', 'entry_end_time',
               'skip_expiry_day', 'eod_squareoff_time', 'time_exit',
               'target_entry_premium', 'min_leg_premium', 'max_leg_premium',
               'min_otm_distance', 'min_squeeze_bars'}
    updated = {}
    for key, val in data.items():
        if key in allowed:
            NAS_DEFAULTS[key] = val
            updated[key] = val
    if not updated:
        return jsonify({'error': 'No valid config keys provided', 'allowed': sorted(allowed)}), 400

    # Update ticker's config copy too
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker()
        for k, v in updated.items():
            ticker.config[k] = v
    except Exception:
        pass

    logger.info(f"[NAS] Config updated: {updated}")
    return jsonify({'updated': updated, 'status': 'OK'})


@app.route('/api/nas/option-chain')
def api_nas_option_chain():
    """Latest option chain snapshots."""
    try:
        from services.nas_db import get_nas_db
        db = get_nas_db()
        return jsonify(db.get_recent_snapshots(limit=100))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- NAS WebSocket Ticker Control ----

@app.route('/api/nas/ticker/start', methods=['POST'])
def api_nas_ticker_start():
    """Start the NAS WebSocket ticker for live 5-min candle streaming."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        ticker.start()
        return jsonify({'status': 'started', 'ticker': ticker.get_status()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/ticker/stop', methods=['POST'])
def api_nas_ticker_stop():
    """Stop the NAS WebSocket ticker."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker()
        ticker.stop()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/ticker/status')
def api_nas_ticker_status():
    """Get NAS ticker connection status."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        return jsonify(ticker.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/ticker/stream')
@app.route('/api/nas/stream')
def api_nas_ticker_stream():
    """SSE stream of live spot + option premiums across ALL NAS systems.

    Single connection per dashboard — v2 React pages should open ONE EventSource
    here. Payload shape:
        {type: 'tick', spot: 24473.25, legs: {TSYM: {ltp, entry, leg}}, ts: ...}
        {type: 'offline'}  — emitted when ticker isn't running
    """
    import json, time as _time

    def generate():
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        last_snapshot = {}
        last_916_fetch = 0.0
        last_916_legs: dict = {}

        # Initial keepalive so clients know the stream is live even before
        # the first meaningful tick arrives.
        yield ": connected\n\n"

        while True:
            ws_alive = ticker.is_running
            legs = {}

            # Squeeze legs come from the WebSocket ticker — only available
            # when the ticker is alive. 9:16 legs ALWAYS use the REST
            # kite.ltp() fallback below, irrespective of ticker state.
            if ws_alive:
                # Merge option tokens from all 4 systems (OTM, ATM, ATM2, ATM4).
                all_tokens = {}
                for attr in ('_option_tokens', '_atm_option_tokens',
                             '_atm2_option_tokens', '_atm4_option_tokens'):
                    all_tokens.update(getattr(ticker, attr, {}) or {})
                all_ltps = {}
                for attr in ('_option_ltps', '_atm_option_ltps',
                             '_atm2_option_ltps', '_atm4_option_ltps'):
                    all_ltps.update(getattr(ticker, attr, {}) or {})

                for token, info in all_tokens.items():
                    tsym = info.get('tradingsymbol')
                    if not tsym:
                        continue
                    ltp = all_ltps.get(token)
                    if ltp is None or ltp <= 0:
                        continue
                    legs[tsym] = {
                        'ltp': round(ltp, 2),
                        'entry': info.get('entry_premium') or info.get('entry_price'),
                        'leg': info.get('leg'),
                    }

            # The WebSocket ticker only subscribes to Squeeze variant legs.
            # 9:16 positions need LTP via REST kite.ltp(). Throttle to ~2s.
            # Run this UNCONDITIONALLY — when ticker is down, this is the
            # only LTP path we have for any active leg, so the dashboard's
            # 9:16 panels still tick.
            now_ts = _time.time()
            if now_ts - last_916_fetch >= 2.0:
                last_916_fetch = now_ts
                try:
                    from services.nas_916_db import (
                        get_nas_916_otm_db, get_nas_916_atm_db,
                        get_nas_916_atm2_db, get_nas_916_atm4_db,
                    )
                    from services.kite_service import get_kite
                    positions_916 = []
                    for _dbf in (get_nas_916_otm_db, get_nas_916_atm_db,
                                 get_nas_916_atm2_db, get_nas_916_atm4_db):
                        try:
                            positions_916.extend(_dbf().get_active_positions() or [])
                        except Exception:
                            continue
                    tsyms = list({p['tradingsymbol'] for p in positions_916
                                  if p.get('tradingsymbol')})
                    ltp_map: dict = {}
                    if tsyms:
                        try:
                            kite = get_kite()
                            q = kite.ltp([f'NFO:{s}' for s in tsyms]) or {}
                            for s in tsyms:
                                v = q.get(f'NFO:{s}')
                                if v and v.get('last_price'):
                                    ltp_map[s] = v['last_price']
                        except Exception:
                            pass
                    last_916_legs = {}
                    for p in positions_916:
                        tsym = p.get('tradingsymbol')
                        if not tsym:
                            continue
                        ltp = ltp_map.get(tsym)
                        if ltp is None or ltp <= 0:
                            continue
                        last_916_legs[tsym] = {
                            'ltp': round(ltp, 2),
                            'entry': p.get('entry_price') or p.get('entry_premium'),
                            'leg': p.get('leg'),
                        }
                except Exception:
                    pass

            # Ticker legs (Squeeze) take precedence; 9:16 legs fill the gap.
            for tsym, info in last_916_legs.items():
                if tsym not in legs:
                    legs[tsym] = info

            # Spot: prefer WebSocket ticker (real-time). When ticker is down,
            # the SSE consumer can fall back to its polled state for spot.
            spot = round(ticker._last_ltp, 2) if (ws_alive and ticker._last_ltp) else 0

            # If ticker is offline AND no 9:16 legs to push, send the
            # offline marker so the client can show a dot. Otherwise push
            # whatever legs we have (REST 9:16 path even if WebSocket dead).
            if not ws_alive and not legs:
                if last_snapshot != {'__offline__': True}:
                    last_snapshot = {'__offline__': True}
                    yield f"data: {json.dumps({'type': 'offline'})}\n\n"
                _time.sleep(5)
                continue

            snapshot = {'spot': spot, 'legs': legs, 'ws_alive': ws_alive}

            # Push only when something changed (plus periodic keepalive)
            if snapshot != last_snapshot:
                last_snapshot = snapshot
                payload = {
                    'type': 'tick',
                    'spot': snapshot['spot'],
                    'legs': snapshot['legs'],
                    'ws_alive': ws_alive,
                    'ts': _time.time(),
                }
                yield f"data: {json.dumps(payload)}\n\n"

            # When ticker is alive, push fast (~3/sec). When only REST
            # 9:16 path is feeding, throttle to one push every ~2s to
            # match the kite.ltp() fetch cadence.
            _time.sleep(0.3 if ws_alive else 2.0)

    return app.response_class(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ═══════════════════════════════════════════════════════════════
# SCC — Strategy Command Center Aggregation APIs
# ═══════════════════════════════════════════════════════════════

STRATEGY_META = {
    'bnf': {'name': 'BNF Squeeze & Fire', 'type': 'F&O', 'capital': BNF_DEFAULTS.get('capital', 1000000)},
    'maruthi': {'name': 'Maruthi Always-On', 'type': 'F&O', 'capital': MARUTHI_DEFAULTS.get('capital', 1500000)},
    'kc6': {'name': 'KC6 Mean Reversion', 'type': 'Equity', 'capital': 400000},
    'nas': {'name': 'NAS OTM Strangle', 'type': 'F&O', 'capital': 300000},
    'nas_atm': {'name': 'NAS ATM Strangle', 'type': 'F&O', 'capital': NAS_ATM_DEFAULTS.get('capital', 500000)},
    'trident': {'name': 'Trident', 'type': 'F&O', 'capital': 1000000},
}

def _scc_get_strategy_state(strategy_id):
    """Get normalized state for a single strategy. Returns dict matching SCC Strategy interface."""
    meta = STRATEGY_META.get(strategy_id, {})
    base = {
        'id': strategy_id,
        'name': meta.get('name', strategy_id),
        'type': meta.get('type', 'Equity'),
        'status': 'stopped',
        'mode': 'paper',
        'capital': meta.get('capital', 0),
        'deployed': 0,
        'pnlToday': 0,
        'pnlTotal': 0,
        'winRate': 0,
        'trades': 0,
        'maxDD': 0,
        'sharpe': 0,
        'lastSignal': 'N/A',
        'risk': 'low',
    }

    try:
        if strategy_id == 'bnf':
            from services.bnf_executor import get_bnf_executor
            executor = get_bnf_executor(BNF_DEFAULTS)
            state = executor.get_full_state()
            enabled = BNF_DEFAULTS.get('enabled', True)
            base['status'] = 'running' if enabled else 'paused'
            base['mode'] = 'live' if not BNF_DEFAULTS.get('paper_trading_mode', True) else 'paper'
            positions = [p for p in state.get('positions', []) if isinstance(p, dict)]
            base['deployed'] = sum(abs(p.get('premium', 0) * p.get('quantity', 0)) for p in positions) if positions else 0
            trades = [t for t in state.get('trades', []) if isinstance(t, dict)]
            base['trades'] = len(trades)
            base['pnlTotal'] = sum(t.get('pnl', t.get('net_pnl', 0)) for t in trades)
            base['pnlToday'] = sum(t.get('pnl', t.get('net_pnl', 0)) for t in trades if t.get('exit_date', t.get('trade_date', '')) == str(date.today()))
            if trades:
                wins = [t for t in trades if t.get('pnl', t.get('net_pnl', 0)) > 0]
                base['winRate'] = round(len(wins) / len(trades) * 100, 1)

        elif strategy_id == 'maruthi':
            from services.maruthi_executor import get_maruthi_executor
            executor = get_maruthi_executor(MARUTHI_DEFAULTS)
            state = executor.get_state()
            enabled = MARUTHI_DEFAULTS.get('enabled', True)
            base['status'] = 'running' if enabled else 'paused'
            base['mode'] = 'live' if not MARUTHI_DEFAULTS.get('paper_trading_mode', True) else 'paper'
            positions = state.get('positions', [])
            base['deployed'] = sum(abs(p.get('value', 0)) for p in positions) if positions else 0
            trades = state.get('trades', [])
            base['trades'] = len(trades)
            base['pnlTotal'] = sum(t.get('pnl', 0) for t in trades)
            base['pnlToday'] = sum(t.get('pnl', 0) for t in trades if t.get('exit_date', '') == str(date.today()))

        elif strategy_id == 'kc6':
            from services.kc6_db import get_kc6_db
            db = get_kc6_db()
            positions = db.get_active_positions()
            stats = db.get_stats()
            enabled = KC6_DEFAULTS.get('enabled', True)
            base['status'] = 'running' if enabled else 'paused'
            base['mode'] = 'live' if not KC6_DEFAULTS.get('paper_trading_mode', True) else 'paper'
            base['deployed'] = sum(p.get('entry_price', 0) * p.get('quantity', 0) for p in positions) if positions else 0
            base['trades'] = stats.get('total_trades', 0)
            base['pnlTotal'] = stats.get('total_pnl', 0)
            base['winRate'] = stats.get('win_rate', 0)
            base['pnlToday'] = stats.get('today_pnl', 0)

        elif strategy_id == 'nas':
            from services.nas_executor import NasExecutor
            executor = NasExecutor(config=NAS_DEFAULTS)
            state = executor.get_full_state()
            enabled = NAS_DEFAULTS.get('enabled', True)
            base['status'] = 'running' if enabled else 'paused'
            base['mode'] = 'live' if not NAS_DEFAULTS.get('paper_trading_mode', True) else 'paper'
            positions = [p for p in state.get('positions', []) if isinstance(p, dict)]
            base['deployed'] = sum(abs(p.get('premium', 0) * p.get('quantity', 0)) for p in positions) if positions else 0
            trades = [t for t in state.get('trades', []) if isinstance(t, dict)]
            base['trades'] = len(trades)
            base['pnlTotal'] = sum(t.get('pnl', t.get('net_pnl', 0)) for t in trades)

        elif strategy_id == 'nas_atm':
            from services.nas_atm_executor import NasAtmExecutor
            executor = NasAtmExecutor(config=NAS_ATM_DEFAULTS)
            state = executor.get_full_state()
            enabled = NAS_ATM_DEFAULTS.get('enabled', True)
            base['status'] = 'running' if enabled else 'paused'
            base['mode'] = 'live' if not NAS_ATM_DEFAULTS.get('paper_trading_mode', True) else 'paper'
            stats = state.get('stats', {})
            base['trades'] = stats.get('total_trades', 0)
            base['pnlTotal'] = stats.get('total_pnl', 0)
            base['winRate'] = stats.get('win_rate', 0)

        elif strategy_id == 'trident':
            from services.trident_executor import get_trident_executor
            executor = get_trident_executor(TRIDENT_CONFIG)
            state = executor.get_state()
            cfg = state.get('config', {})
            base['status'] = 'running' if cfg.get('enabled', True) else 'paused'
            base['mode'] = 'live' if not cfg.get('paper_trading_mode', True) else 'paper'
            stats = state.get('stats', {})
            base['trades'] = stats.get('total_trades', 0)
            base['pnlTotal'] = stats.get('total_pnl', 0)
            base['winRate'] = stats.get('win_rate', 0)
            base['deployed'] = sum(p.get('entry_price', 0) * p.get('qty', 0) for p in state.get('positions', []))

    except Exception as e:
        logger.warning(f"[SCC] Error fetching {strategy_id} state: {e}")
        base['status'] = 'error'
        base['lastSignal'] = str(e)[:50]

    return base


def _scc_get_positions(strategy_id):
    """Get normalized positions for a strategy. Returns list matching SCC Position interface."""
    positions = []
    try:
        if strategy_id == 'bnf':
            from services.bnf_executor import get_bnf_executor
            executor = get_bnf_executor(BNF_DEFAULTS)
            state = executor.get_full_state()
            for i, p in enumerate(state.get('positions', [])):
                positions.append({
                    'id': f'bnf_{i}',
                    'strategy': 'BNF Squeeze & Fire',
                    'symbol': p.get('tradingsymbol', p.get('symbol', 'BANKNIFTY')),
                    'type': 'OPT',
                    'side': p.get('side', 'SHORT').upper(),
                    'qty': abs(p.get('quantity', 0)),
                    'entry': p.get('entry_price', p.get('premium', 0)),
                    'current': p.get('ltp', p.get('entry_price', 0)),
                    'pnl': p.get('pnl', 0),
                    'pnlPct': p.get('pnl_pct', 0),
                    'sl': p.get('sl', 0),
                    'tp': p.get('tp', None),
                    'flags': [],
                })

        elif strategy_id == 'maruthi':
            from services.maruthi_executor import get_maruthi_executor
            executor = get_maruthi_executor(MARUTHI_DEFAULTS)
            state = executor.get_state()
            for i, p in enumerate(state.get('positions', [])):
                inst_type = 'FUT' if 'FUT' in str(p.get('tradingsymbol', '')).upper() else 'OPT'
                positions.append({
                    'id': f'maruthi_{i}',
                    'strategy': 'Maruthi Always-On',
                    'symbol': p.get('tradingsymbol', 'MARUTI FUT'),
                    'type': inst_type,
                    'side': p.get('side', 'LONG').upper(),
                    'qty': abs(p.get('quantity', 0)),
                    'entry': p.get('entry_price', 0),
                    'current': p.get('ltp', p.get('entry_price', 0)),
                    'pnl': p.get('pnl', 0),
                    'pnlPct': p.get('pnl_pct', 0),
                    'sl': p.get('sl', 0),
                    'tp': p.get('tp', None),
                    'flags': [],
                })

        elif strategy_id == 'kc6':
            from services.kc6_db import get_kc6_db
            db = get_kc6_db()
            for i, p in enumerate(db.get_active_positions()):
                entry = p.get('entry_price', 0)
                current = p.get('ltp', entry)
                qty = p.get('quantity', 0)
                pnl = (current - entry) * qty
                pnl_pct = ((current - entry) / entry * 100) if entry else 0
                sl = entry * (1 - KC6_DEFAULTS.get('sl_pct', 5.0) / 100)
                tp = entry * (1 + KC6_DEFAULTS.get('tp_pct', 15.0) / 100)
                flags = []
                if pnl < -4000:
                    flags.append(f'Loss > ₹{abs(pnl)/1000:.0f}K')
                positions.append({
                    'id': f'kc6_{i}',
                    'strategy': 'KC6 Mean Reversion',
                    'symbol': p.get('symbol', ''),
                    'type': 'EQ',
                    'side': 'LONG',
                    'qty': qty,
                    'entry': round(entry, 2),
                    'current': round(current, 2),
                    'pnl': round(pnl, 2),
                    'pnlPct': round(pnl_pct, 2),
                    'sl': round(sl, 2),
                    'tp': round(tp, 2),
                    'flags': flags,
                })

        elif strategy_id == 'nas':
            from services.nas_executor import NasExecutor
            executor = NasExecutor(config=NAS_DEFAULTS)
            state = executor.get_full_state()
            for i, p in enumerate(state.get('positions', [])):
                positions.append({
                    'id': f'nas_{i}',
                    'strategy': 'NAS OTM Strangle',
                    'symbol': p.get('tradingsymbol', p.get('symbol', 'NIFTY')),
                    'type': 'OPT',
                    'side': p.get('side', 'SHORT').upper(),
                    'qty': abs(p.get('quantity', 0)),
                    'entry': p.get('entry_price', p.get('premium', 0)),
                    'current': p.get('ltp', p.get('entry_price', 0)),
                    'pnl': p.get('pnl', 0),
                    'pnlPct': p.get('pnl_pct', 0),
                    'sl': p.get('sl', 0),
                    'tp': p.get('tp', None),
                    'flags': [],
                })

        elif strategy_id == 'nas_atm':
            from services.nas_atm_executor import NasAtmExecutor
            executor = NasAtmExecutor(config=NAS_ATM_DEFAULTS)
            state = executor.get_full_state()
            all_pos = (state.get('positions', {}).get('ce', []) +
                       state.get('positions', {}).get('pe', []))
            for i, p in enumerate(all_pos):
                positions.append({
                    'id': f'nas_atm_{i}',
                    'strategy': 'NAS ATM Strangle',
                    'symbol': p.get('tradingsymbol', 'NIFTY'),
                    'type': 'OPT',
                    'side': 'SHORT',
                    'qty': abs(p.get('qty', 0)),
                    'entry': p.get('entry_price', 0),
                    'current': p.get('entry_price', 0),
                    'pnl': 0,
                    'pnlPct': 0,
                    'sl': p.get('sl_price', 0),
                    'tp': None,
                    'flags': [],
                })

    except Exception as e:
        logger.warning(f"[SCC] Error fetching {strategy_id} positions: {e}")

    return positions


def _scc_get_trades(strategy_id, limit=50):
    """Get normalized closed trades for a strategy. Returns list matching SCC Trade interface."""
    trades = []
    try:
        if strategy_id == 'bnf':
            from services.bnf_db import get_bnf_db
            db = get_bnf_db()
            for t in db.get_recent_trades(limit=limit):
                trades.append({
                    'id': f"bnf_t_{t.get('id', '')}",
                    'date': t.get('exit_date', t.get('date', '')),
                    'strategy': 'BNF Squeeze & Fire',
                    'symbol': t.get('tradingsymbol', t.get('symbol', '')),
                    'side': t.get('side', 'SHORT').upper(),
                    'entry': t.get('entry_price', 0),
                    'exit': t.get('exit_price', 0),
                    'pnl': t.get('pnl', 0),
                    'pnlPct': t.get('pnl_pct', 0),
                    'holding': t.get('holding_period', ''),
                    'notes': t.get('exit_reason', ''),
                    'journal': '',
                })

        elif strategy_id == 'kc6':
            from services.kc6_db import get_kc6_db
            db = get_kc6_db()
            for t in db.get_trade_history(limit=limit):
                trades.append({
                    'id': f"kc6_t_{t.get('id', '')}",
                    'date': t.get('exit_date', t.get('date', '')),
                    'strategy': 'KC6 Mean Reversion',
                    'symbol': t.get('symbol', ''),
                    'side': 'LONG',
                    'entry': t.get('entry_price', 0),
                    'exit': t.get('exit_price', 0),
                    'pnl': t.get('pnl', 0),
                    'pnlPct': round(((t.get('exit_price', 0) - t.get('entry_price', 1)) / t.get('entry_price', 1) * 100), 2) if t.get('entry_price') else 0,
                    'holding': t.get('holding_days', ''),
                    'notes': t.get('exit_reason', ''),
                    'journal': '',
                })

        elif strategy_id == 'nas':
            from services.nas_db import get_nas_db
            db = get_nas_db()
            for t in db.get_recent_trades(limit=limit):
                trades.append({
                    'id': f"nas_t_{t.get('id', '')}",
                    'date': t.get('exit_date', t.get('date', '')),
                    'strategy': 'NAS OTM Strangle',
                    'symbol': t.get('tradingsymbol', t.get('symbol', '')),
                    'side': t.get('side', 'SHORT').upper(),
                    'entry': t.get('entry_price', 0),
                    'exit': t.get('exit_price', 0),
                    'pnl': t.get('pnl', 0),
                    'pnlPct': t.get('pnl_pct', 0),
                    'holding': t.get('holding_period', ''),
                    'notes': t.get('exit_reason', ''),
                    'journal': '',
                })

        elif strategy_id == 'nas_atm':
            from services.nas_atm_db import get_nas_atm_db
            db = get_nas_atm_db()
            for t in db.get_recent_trades(limit=limit):
                trades.append({
                    'id': f"nas_atm_t_{t.get('id', '')}",
                    'date': t.get('trade_date', ''),
                    'strategy': 'NAS ATM Strangle',
                    'symbol': 'NIFTY',
                    'side': 'SHORT',
                    'entry': t.get('total_premium_collected', 0),
                    'exit': t.get('total_premium_paid', 0),
                    'pnl': t.get('net_pnl', 0),
                    'pnlPct': 0,
                    'holding': 'Intraday',
                    'notes': t.get('exit_reason', ''),
                    'journal': '',
                })

    except Exception as e:
        logger.warning(f"[SCC] Error fetching {strategy_id} trades: {e}")

    return trades


@app.route('/api/scc/dashboard')
def api_scc_dashboard():
    """Aggregated dashboard: all strategy states + positions + P&L."""
    strategies = []
    all_positions = []

    for sid in STRATEGY_META:
        strategies.append(_scc_get_strategy_state(sid))
        all_positions.extend(_scc_get_positions(sid))

    total_capital = sum(s['capital'] for s in strategies)
    total_deployed = sum(s['deployed'] for s in strategies)
    today_pnl = sum(s['pnlToday'] for s in strategies)
    total_pnl = sum(s['pnlTotal'] for s in strategies)

    return jsonify({
        'strategies': strategies,
        'positions': all_positions,
        'summary': {
            'totalCapital': total_capital,
            'totalDeployed': total_deployed,
            'todayPnl': today_pnl,
            'totalPnl': total_pnl,
            'runningCount': len([s for s in strategies if s['status'] == 'running']),
            'positionCount': len(all_positions),
        },
    })


@app.route('/api/scc/positions')
def api_scc_positions():
    """All open positions across all strategies."""
    all_positions = []
    for sid in STRATEGY_META:
        all_positions.extend(_scc_get_positions(sid))
    return jsonify(all_positions)


@app.route('/api/scc/trades')
def api_scc_trades():
    """All closed trades, optionally filtered by strategy and win/loss."""
    strategy = request.args.get('strategy', 'all')
    filt = request.args.get('filter', 'all')

    all_trades = []
    sids = [strategy] if strategy != 'all' and strategy in STRATEGY_META else list(STRATEGY_META.keys())
    for sid in sids:
        all_trades.extend(_scc_get_trades(sid))

    # Sort by date descending
    all_trades.sort(key=lambda t: t.get('date', ''), reverse=True)

    if filt == 'wins':
        all_trades = [t for t in all_trades if t['pnl'] > 0]
    elif filt == 'losses':
        all_trades = [t for t in all_trades if t['pnl'] <= 0]

    return jsonify(all_trades[:100])


@app.route('/api/scc/kill-all', methods=['POST'])
def api_scc_kill_all():
    """Emergency: kill-switch all strategies and close all positions."""
    results = {}
    kill_funcs = {
        'bnf': lambda: __import__('services.bnf_executor', fromlist=['get_bnf_executor']).get_bnf_executor(BNF_DEFAULTS).emergency_exit_all(),
        'kc6': lambda: KC6_DEFAULTS.update({'enabled': False}),
        'maruthi': lambda: MARUTHI_DEFAULTS.update({'enabled': False}),
        'nas': lambda: NAS_DEFAULTS.update({'enabled': False}),
        'nas_atm': lambda: NAS_ATM_DEFAULTS.update({'enabled': False}),
    }
    for sid, fn in kill_funcs.items():
        try:
            fn()
            results[sid] = 'killed'
        except Exception as e:
            results[sid] = f'error: {str(e)}'
            logger.error(f"[SCC] Kill-all error for {sid}: {e}")

    return jsonify({'status': 'EMERGENCY_KILL_ALL', 'results': results})


@app.route('/api/scc/toggle-strategy', methods=['POST'])
def api_scc_toggle_strategy():
    """Toggle a strategy enabled/disabled."""
    data = request.get_json() or {}
    sid = data.get('id', '')

    config_map = {
        'bnf': BNF_DEFAULTS,
        'maruthi': MARUTHI_DEFAULTS,
        'kc6': KC6_DEFAULTS,
        'nas': NAS_DEFAULTS,
        'nas_atm': NAS_ATM_DEFAULTS,
    }

    if sid not in config_map:
        return jsonify({'error': f'Unknown strategy: {sid}'}), 400

    cfg = config_map[sid]
    current = cfg.get('enabled', True)
    cfg['enabled'] = not current
    new_status = 'running' if cfg['enabled'] else 'paused'
    logger.info(f"[SCC] Strategy {sid} toggled to {new_status}")
    return jsonify({'id': sid, 'status': new_status, 'enabled': cfg['enabled']})


@app.route('/api/scc/blueprints')
def api_scc_blueprints():
    """Static strategy blueprints — served from frontend data for now."""
    return jsonify({'status': 'use_frontend_data'})


# ---- NAS Scheduled Jobs ----
# NOTE: Entry scanning and position monitoring are handled by the WebSocket ticker
# (services/nas_ticker.py). Only time-based jobs remain here as cron jobs.

def _nas_ticker_autostart():
    """9:16 AM Mon-Wed,Fri — Auto-start NAS WebSocket ticker after market open."""
    if not NAS_DEFAULTS.get('enabled', True):
        return
    if NAS_DEFAULTS.get('skip_expiry_day', True) and date.today().weekday() == 3:
        return
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        if not ticker.is_connected:
            ticker.restart()
            logger.info("[NAS] Ticker auto-started at 9:16 AM")
        else:
            logger.info("[NAS] Ticker already connected")
    except Exception as e:
        logger.error(f"[NAS] Ticker auto-start error: {e}")


def _nas_eod_squareoff():
    """3:15 PM Mon-Wed,Fri — Mandatory EOD squareoff."""
    if not NAS_DEFAULTS.get('enabled', True):
        return
    if NAS_DEFAULTS.get('skip_expiry_day', True) and date.today().weekday() == 3:
        return
    try:
        from services.nas_executor import NasExecutor
        executor = NasExecutor(config=NAS_DEFAULTS)
        exits = executor.eod_squareoff()
        logger.info(f"[NAS] EOD squareoff: {len(exits)} positions closed")

        # Unsubscribe option legs after EOD
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker()
        ticker.subscribe_option_legs([])
    except Exception as e:
        logger.error(f"[NAS] EOD squareoff error: {e}")


def _nas_market_close():
    """3:30 PM Mon-Fri — Force-close current candle, stop ticker."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        ticker.aggregator.force_close()
        logger.info("[NAS] Forced final candle close at market close")
    except Exception as e:
        logger.error(f"[NAS] Market close handler failed: {e}")


def _nas_daily_summary():
    """3:20 PM Mon-Wed,Fri — Save daily summary."""
    if not NAS_DEFAULTS.get('enabled', True):
        return
    try:
        from services.nas_db import get_nas_db
        db = get_nas_db()
        stats = db.get_stats()
        logger.info(f"[NAS] Daily summary: {stats.get('total_trades', 0)} total trades, "
                     f"P&L Rs {stats.get('total_pnl', 0):.0f}")
    except Exception as e:
        logger.error(f"[NAS] Daily summary error: {e}")


try:
    # Auto-start ticker at 9:16 AM (after market open)
    # NOTE 2026-04-23: was 'mon-wed,fri' (skip NIFTY weekly expiry Thursday).
    # Switched to 'mon-fri' — since these are paper-mode strategies, user
    # wants Thursday data captured too for apples-to-apples comparison.
    scheduler.add_job(
        _nas_ticker_autostart,
        'cron', day_of_week='mon-fri', hour=9, minute=16,
        id='nas_ticker_autostart', replace_existing=True,
    )
    # EOD squareoff at 3:15 PM
    scheduler.add_job(
        _nas_eod_squareoff,
        'cron', day_of_week='mon-fri', hour=15, minute=15,
        id='nas_eod_squareoff', replace_existing=True,
    )
    # Force close candle at 3:30 PM
    scheduler.add_job(
        _nas_market_close,
        'cron', day_of_week='mon-fri', hour=15, minute=30,
        id='nas_market_close', replace_existing=True,
    )
    # Daily summary at 3:20 PM
    scheduler.add_job(
        _nas_daily_summary,
        'cron', day_of_week='mon-fri', hour=15, minute=20,
        id='nas_daily_summary', replace_existing=True,
    )
    logger.info(
        "NAS scheduled jobs registered: "
        "ticker autostart(9:16), EOD squareoff(15:15), "
        "market close(15:30), daily summary(15:20) — Mon-Fri"
    )
except Exception as e:
    logger.warning(f"Could not register NAS scheduled jobs: {e}")


# DB integrity watchdog — checks all NAS + ORB DBs every 5 min during market
# hours. Born out of the 2026-04-24 NAS table-vanish incident. See
# services/db_watchdog.py for the watched list + alert dispatch.
try:
    from services.db_watchdog import run_watchdog_check
    scheduler.add_job(
        run_watchdog_check,
        'cron', day_of_week='mon-fri', hour='9-15', minute='*/5',
        id='db_integrity_watchdog', replace_existing=True,
    )
    logger.info("DB integrity watchdog scheduled: every 5 min Mon-Fri 9-15 IST")
except Exception as e:
    logger.warning(f"Could not register db_integrity_watchdog: {e}")


# NAS ticker watchdog — checks every 5 min during market hours that the
# NAS WebSocket ticker is alive. If is_running OR is_connected is False,
# restarts it. Born out of the 2026-04-28 incident: ticker died ~09:19 IST
# after 9:16 entries+SL hits, no auto-recovery, killed Squeeze ATR
# detection (zero entries despite squeeze conditions) and froze live LTP
# feed for 9:16 open legs. The 5-min cadence gives Kite time to clear stale
# WebSocket slots between attempts, avoiding reconnect storms.
def _nas_ticker_watchdog():
    """Restart NAS ticker if down. Only attempts during 09:15-15:25 IST
    (wraps the trading window with a small safety margin)."""
    now_t = datetime.now().time()
    if now_t < dtime(9, 15) or now_t >= dtime(15, 25):
        return
    try:
        from services.nas_ticker import get_nas_ticker, stop_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        if ticker.is_running and ticker.is_connected:
            return  # healthy
        # Ticker is down — log + nuke the singleton + start fresh.
        # stop_nas_ticker() sets _nas_ticker = None so the next get_nas_ticker
        # creates a brand-new instance, dodging zombie-WebSocket state from
        # the prior dead instance.
        logger.warning(
            f"[NAS-WATCHDOG] Ticker down "
            f"(running={ticker.is_running}, connected={ticker.is_connected}) "
            f"— issuing stop + fresh start"
        )
        try:
            ticker.stop()
        except Exception as e:
            logger.warning(f"[NAS-WATCHDOG] stop failed (ignoring): {e}")
        stop_nas_ticker()  # clears the singleton
        fresh = get_nas_ticker(NAS_DEFAULTS)
        fresh.start()
        logger.info("[NAS-WATCHDOG] Restart issued")
    except Exception as e:
        logger.error(f"[NAS-WATCHDOG] Error: {e}", exc_info=True)


try:
    scheduler.add_job(
        _nas_ticker_watchdog,
        'cron', day_of_week='mon-fri', hour='9-15', minute='*/5',
        id='nas_ticker_watchdog', replace_existing=True,
    )
    logger.info("NAS ticker watchdog scheduled: every 5 min Mon-Fri 09:15-15:25 IST")
except Exception as e:
    logger.warning(f"Could not register nas_ticker_watchdog: {e}")


# System validator — pre-market checklist (08:50 IST) + EOD report (15:40 IST).
# Emails the operator and stores in-app. Covers ORB cash, NAS x8, ORB index.
try:
    from services.system_validator import run_premarket_check, run_eod_check
    scheduler.add_job(
        run_premarket_check,
        'cron', day_of_week='mon-fri', hour=8, minute=50,
        id='system_validator_premarket', replace_existing=True,
    )
    scheduler.add_job(
        run_eod_check,
        'cron', day_of_week='mon-fri', hour=15, minute=40,
        id='system_validator_eod', replace_existing=True,
    )
    logger.info("System validator scheduled: pre-market(08:50), EOD(15:40) Mon-Fri IST")
except Exception as e:
    logger.warning(f"Could not register system_validator jobs: {e}")


# Pre-market brief — 2-stage flow:
#   08:00 IST: VPS builds raw data (yfinance + RSS + holdings + F&O ban)
#   08:02 IST: Cloud Claude routine fetches /api/premarket/brief/raw,
#              synthesizes headlines + narrative, POSTs to /synthesized
#              which dispatches the email
#   08:08 IST: VPS fallback — if no synthesized email went out yet, sends
#              the un-synthesized version (so operator always gets one)
try:
    from services.premarket_brief import (
        run_premarket_brief_build_only, run_premarket_brief_fallback
    )
    scheduler.add_job(
        run_premarket_brief_build_only,
        'cron', day_of_week='mon-fri', hour=8, minute=0,
        id='premarket_brief_build', replace_existing=True,
    )
    scheduler.add_job(
        run_premarket_brief_fallback,
        'cron', day_of_week='mon-fri', hour=8, minute=8,
        id='premarket_brief_fallback', replace_existing=True,
    )
    logger.info("Pre-market brief scheduled: build(08:00) + fallback(08:08) Mon-Fri IST")
except Exception as e:
    logger.warning(f"Could not register premarket_brief jobs: {e}")


# =============================================================================
# NAS ATM — Nifty ATM Strangle (Cascading, per-leg SL)
# =============================================================================

_nas_atm_tasks = {}
_nas_atm2_tasks = {}
_nas_atm4_tasks = {}


@app.route('/nas-atm')
def nas_atm_dashboard():
    """NAS ATM Dashboard — redirects to combined NAS page."""
    return redirect(url_for('nas_dashboard'))


@app.route('/api/nas-atm/state')
def api_nas_atm_state():
    """Full state dump for NAS ATM dashboard."""
    try:
        from services.nas_atm_executor import NasAtmExecutor
        executor = NasAtmExecutor(config=NAS_ATM_DEFAULTS)
        state = executor.get_full_state()
        state = _enrich_nas_positions_with_ltp(state, '_atm_option_ltps', '_atm_option_tokens')
        return jsonify(state)
    except Exception as e:
        logger.error(f"NAS-ATM state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm/scan', methods=['POST'])
def api_nas_atm_scan():
    """Trigger a manual NAS ATM scan (entry attempt)."""
    task_id = f"nas_atm_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _nas_atm_tasks[task_id] = {'status': 'running'}

    def _run_scan(tid):
        try:
            from services.nas_atm_executor import NasAtmExecutor
            executor = NasAtmExecutor(config=NAS_ATM_DEFAULTS)
            sid, msg = executor.execute_strangle_entry()
            _nas_atm_tasks[tid] = {'status': 'completed', 'result': {'strangle_id': sid, 'message': msg}}
        except Exception as e:
            logger.error(f"NAS-ATM scan error: {e}")
            _nas_atm_tasks[tid] = {'status': 'error', 'error': str(e)}

    scheduler.add_job(_run_scan, args=[task_id], id=f'nas_atm_scan_{task_id}',
                      replace_existing=True)
    return jsonify({'task_id': task_id, 'status': 'queued'})


@app.route('/api/nas-atm/scan/status/<task_id>')
def api_nas_atm_scan_status(task_id):
    """Poll NAS ATM scan status."""
    task = _nas_atm_tasks.get(task_id, {'status': 'unknown'})
    return jsonify(task)


@app.route('/api/nas-atm/kill-switch', methods=['POST'])
def api_nas_atm_kill_switch():
    """Emergency close all NAS ATM positions."""
    try:
        from services.nas_atm_executor import NasAtmExecutor
        executor = NasAtmExecutor(config=NAS_ATM_DEFAULTS)
        exits = executor.emergency_exit_all()
        return jsonify({'status': 'killed', 'positions_closed': exits})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm/trades')
def api_nas_atm_trades():
    """Recent NAS ATM trades."""
    try:
        from services.nas_atm_db import get_nas_atm_db
        db = get_nas_atm_db()
        return jsonify(db.get_recent_trades(limit=50))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm/signals')
def api_nas_atm_signals():
    """NAS ATM signal history."""
    try:
        from services.nas_atm_db import get_nas_atm_db
        db = get_nas_atm_db()
        return jsonify(db.get_recent_signals(limit=30))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm/equity-curve')
def api_nas_atm_equity_curve():
    """NAS ATM daily P&L for equity curve chart."""
    try:
        from services.nas_atm_db import get_nas_atm_db
        db = get_nas_atm_db()
        return jsonify(db.get_equity_curve())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm/toggle-mode', methods=['POST'])
def api_nas_atm_toggle_mode():
    """Toggle NAS ATM paper/live trading mode."""
    try:
        current = NAS_ATM_DEFAULTS.get('paper_trading_mode', True)
        NAS_ATM_DEFAULTS['paper_trading_mode'] = not current
        new_mode = 'PAPER' if NAS_ATM_DEFAULTS['paper_trading_mode'] else 'LIVE'
        logger.info(f"NAS-ATM mode toggled to {new_mode}")
        return jsonify({'mode': new_mode})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm/toggle-enabled', methods=['POST'])
def api_nas_atm_toggle_enabled():
    """Enable/disable NAS ATM system."""
    try:
        current = NAS_ATM_DEFAULTS.get('enabled', True)
        NAS_ATM_DEFAULTS['enabled'] = not current
        new_enabled = NAS_ATM_DEFAULTS['enabled']
        status = 'enabled' if new_enabled else 'disabled'
        logger.info(f"NAS-ATM system {status}")
        return jsonify({'enabled': new_enabled, 'status': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm/ticker/start', methods=['POST'])
def api_nas_atm_ticker_start():
    """Start NAS ticker (shared with OTM) for ATM system."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        if not ticker.is_running:
            ticker.start()
        return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm/ticker/stop', methods=['POST'])
def api_nas_atm_ticker_stop():
    """Stop NAS ticker (shared)."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker()
        ticker.stop()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm/ticker/status')
def api_nas_atm_ticker_status():
    """Get NAS ticker status (shared)."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        return jsonify(ticker.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm/ticker/stream')
def api_nas_atm_ticker_stream():
    """SSE stream for NAS ATM tick-by-tick updates (shares NIFTY spot from NAS ticker)."""
    import time as _time

    def generate():
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        last_spot = 0
        while True:
            _time.sleep(1)
            if not ticker.is_running:
                yield f"data: {json.dumps({'type': 'status', 'running': False})}\n\n"
                continue
            spot = ticker._last_ltp
            if spot != last_spot and spot > 0:
                last_spot = spot
                # Include ATM option leg premiums
                atm_legs = {}
                for token, info in ticker._atm_option_tokens.items():
                    tsym = info['tradingsymbol']
                    ltp = ticker._atm_option_ltps.get(token)
                    if ltp is not None:
                        atm_legs[tsym] = {'ltp': ltp, 'sl': info.get('sl_price', 0)}
                yield f"data: {json.dumps({'type': 'tick', 'spot': round(spot, 2), 'legs': atm_legs})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ---- NAS ATM Scheduled Jobs ----

def _nas_atm_eod_squareoff():
    """3:15 PM — EOD squareoff for NAS ATM."""
    if not NAS_ATM_DEFAULTS.get('enabled', True):
        return
    try:
        from services.nas_atm_executor import NasAtmExecutor
        executor = NasAtmExecutor(config=NAS_ATM_DEFAULTS)
        exits = executor.eod_squareoff()
        logger.info(f"[NAS-ATM] EOD squareoff: {len(exits)} positions closed")

        # Unsubscribe ATM legs from ticker
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker()
        ticker.subscribe_atm_option_legs([])
    except Exception as e:
        logger.error(f"[NAS-ATM] EOD squareoff error: {e}")


def _nas_atm_daily_summary():
    """3:20 PM — Daily summary for NAS ATM."""
    if not NAS_ATM_DEFAULTS.get('enabled', True):
        return
    try:
        from services.nas_atm_db import get_nas_atm_db
        db = get_nas_atm_db()
        stats = db.get_stats()
        logger.info(f"[NAS-ATM] Daily summary: {stats.get('total_trades', 0)} total trades, "
                     f"P&L: Rs {stats.get('total_pnl', 0):.0f}")
    except Exception as e:
        logger.error(f"[NAS-ATM] Daily summary error: {e}")


try:
    scheduler.add_job(
        _nas_atm_eod_squareoff,
        'cron', hour=15, minute=15, day_of_week='mon-fri',
        id='nas_atm_eod_squareoff', replace_existing=True,
    )
    scheduler.add_job(
        _nas_atm_daily_summary,
        'cron', hour=15, minute=20, day_of_week='mon-fri',
        id='nas_atm_daily_summary', replace_existing=True,
    )
    logger.info(
        "NAS ATM scheduled jobs registered: EOD squareoff(15:15), daily summary(15:20)"
    )
except Exception as e:
    logger.warning(f"Could not register NAS ATM scheduled jobs: {e}")


# =============================================================================
# NAS ATM2 — Nifty ATM Strangle V2 (Variant config)
# =============================================================================


@app.route('/api/nas-atm2/state')
def api_nas_atm2_state():
    """Full state dump for NAS ATM2 dashboard."""
    try:
        from services.nas_atm2_executor import NasAtm2Executor
        executor = NasAtm2Executor(config=NAS_ATM2_DEFAULTS)
        state = executor.get_full_state()
        state = _enrich_nas_positions_with_ltp(state, '_atm2_option_ltps', '_atm2_option_tokens')
        return jsonify(state)
    except Exception as e:
        logger.error(f"[NAS-ATM2] state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm2/scan', methods=['POST'])
def api_nas_atm2_scan():
    """Trigger a manual NAS ATM2 scan (entry attempt)."""
    task_id = f"nas_atm2_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _nas_atm2_tasks[task_id] = {'status': 'running'}

    def _run_scan(tid):
        try:
            from services.nas_atm2_executor import NasAtm2Executor
            executor = NasAtm2Executor(config=NAS_ATM2_DEFAULTS)
            sid, msg = executor.execute_strangle_entry()
            _nas_atm2_tasks[tid] = {'status': 'completed', 'result': {'strangle_id': sid, 'message': msg}}
        except Exception as e:
            logger.error(f"[NAS-ATM2] scan error: {e}")
            _nas_atm2_tasks[tid] = {'status': 'error', 'error': str(e)}

    scheduler.add_job(_run_scan, args=[task_id], id=f'nas_atm2_scan_{task_id}',
                      replace_existing=True)
    return jsonify({'task_id': task_id, 'status': 'queued'})


@app.route('/api/nas-atm2/scan/status/<task_id>')
def api_nas_atm2_scan_status(task_id):
    """Poll NAS ATM2 scan status."""
    task = _nas_atm2_tasks.get(task_id, {'status': 'unknown'})
    return jsonify(task)


@app.route('/api/nas-atm2/kill-switch', methods=['POST'])
def api_nas_atm2_kill_switch():
    """Emergency close all NAS ATM2 positions."""
    try:
        from services.nas_atm2_executor import NasAtm2Executor
        executor = NasAtm2Executor(config=NAS_ATM2_DEFAULTS)
        exits = executor.emergency_exit_all()
        return jsonify({'status': 'killed', 'positions_closed': exits})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm2/trades')
def api_nas_atm2_trades():
    """Recent NAS ATM2 trades."""
    try:
        from services.nas_atm2_db import get_nas_atm2_db
        db = get_nas_atm2_db()
        return jsonify(db.get_recent_trades(limit=50))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm2/signals')
def api_nas_atm2_signals():
    """NAS ATM2 signal history."""
    try:
        from services.nas_atm2_db import get_nas_atm2_db
        db = get_nas_atm2_db()
        return jsonify(db.get_recent_signals(limit=30))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm2/equity-curve')
def api_nas_atm2_equity_curve():
    """NAS ATM2 daily P&L for equity curve chart."""
    try:
        from services.nas_atm2_db import get_nas_atm2_db
        db = get_nas_atm2_db()
        return jsonify(db.get_equity_curve())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm2/toggle-mode', methods=['POST'])
def api_nas_atm2_toggle_mode():
    """Toggle NAS ATM2 paper/live trading mode."""
    try:
        current = NAS_ATM2_DEFAULTS.get('paper_trading_mode', True)
        NAS_ATM2_DEFAULTS['paper_trading_mode'] = not current
        new_mode = 'PAPER' if NAS_ATM2_DEFAULTS['paper_trading_mode'] else 'LIVE'
        logger.info(f"[NAS-ATM2] mode toggled to {new_mode}")
        return jsonify({'mode': new_mode})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm2/toggle-enabled', methods=['POST'])
def api_nas_atm2_toggle_enabled():
    """Enable/disable NAS ATM2 system."""
    try:
        current = NAS_ATM2_DEFAULTS.get('enabled', True)
        NAS_ATM2_DEFAULTS['enabled'] = not current
        new_enabled = NAS_ATM2_DEFAULTS['enabled']
        status = 'enabled' if new_enabled else 'disabled'
        logger.info(f"[NAS-ATM2] system {status}")
        return jsonify({'enabled': new_enabled, 'status': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm2/ticker/start', methods=['POST'])
def api_nas_atm2_ticker_start():
    """Start NAS ticker (shared with OTM/ATM) for ATM2 system."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        if not ticker.is_running:
            ticker.start()
        return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm2/ticker/stop', methods=['POST'])
def api_nas_atm2_ticker_stop():
    """Stop NAS ticker (shared)."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker()
        ticker.stop()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm2/ticker/status')
def api_nas_atm2_ticker_status():
    """Get NAS ticker status (shared)."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        return jsonify(ticker.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm2/ticker/stream')
def api_nas_atm2_ticker_stream():
    """SSE stream for NAS ATM2 tick-by-tick updates."""
    import time as _time

    def generate():
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        last_spot = 0
        while True:
            _time.sleep(1)
            if not ticker.is_running:
                yield f"data: {json.dumps({'type': 'status', 'running': False})}\n\n"
                continue
            spot = ticker._last_ltp
            if spot != last_spot and spot > 0:
                last_spot = spot
                atm2_legs = {}
                for token, info in ticker._atm2_option_tokens.items():
                    tsym = info['tradingsymbol']
                    ltp = ticker._atm2_option_ltps.get(token)
                    if ltp is not None:
                        atm2_legs[tsym] = {'ltp': ltp, 'sl': info.get('sl_price', 0)}
                yield f"data: {json.dumps({'type': 'tick', 'spot': round(spot, 2), 'legs': atm2_legs})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ---- NAS ATM2 Scheduled Jobs ----

def _nas_atm2_eod_squareoff():
    """3:15 PM — EOD squareoff for NAS ATM2."""
    if not NAS_ATM2_DEFAULTS.get('enabled', True):
        return
    try:
        from services.nas_atm2_executor import NasAtm2Executor
        executor = NasAtm2Executor(config=NAS_ATM2_DEFAULTS)
        exits = executor.eod_squareoff()
        logger.info(f"[NAS-ATM2] EOD squareoff: {len(exits)} positions closed")

        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker()
        ticker.subscribe_atm2_option_legs([])
    except Exception as e:
        logger.error(f"[NAS-ATM2] EOD squareoff error: {e}")


def _nas_atm2_daily_summary():
    """3:20 PM — Daily summary for NAS ATM2."""
    if not NAS_ATM2_DEFAULTS.get('enabled', True):
        return
    try:
        from services.nas_atm2_db import get_nas_atm2_db
        db = get_nas_atm2_db()
        stats = db.get_stats()
        logger.info(f"[NAS-ATM2] Daily summary: {stats.get('total_trades', 0)} total trades, "
                     f"P&L: Rs {stats.get('total_pnl', 0):.0f}")
    except Exception as e:
        logger.error(f"[NAS-ATM2] Daily summary error: {e}")


try:
    scheduler.add_job(
        _nas_atm2_eod_squareoff,
        'cron', hour=15, minute=15, day_of_week='mon-fri',
        id='nas_atm2_eod_squareoff', replace_existing=True,
    )
    scheduler.add_job(
        _nas_atm2_daily_summary,
        'cron', hour=15, minute=20, day_of_week='mon-fri',
        id='nas_atm2_daily_summary', replace_existing=True,
    )
    logger.info(
        "NAS ATM2 scheduled jobs registered: EOD squareoff(15:15), daily summary(15:20)"
    )
except Exception as e:
    logger.warning(f"Could not register NAS ATM2 scheduled jobs: {e}")


# =============================================================================
# NAS ATM4 — Nifty ATM Strangle V4 (Variant config)
# =============================================================================


@app.route('/api/nas-atm4/state')
def api_nas_atm4_state():
    """Full state dump for NAS ATM4 dashboard."""
    try:
        from services.nas_atm4_executor import NasAtm4Executor
        executor = NasAtm4Executor(config=NAS_ATM4_DEFAULTS)
        state = executor.get_full_state()
        state = _enrich_nas_positions_with_ltp(state, '_atm4_option_ltps', '_atm4_option_tokens')
        return jsonify(state)
    except Exception as e:
        logger.error(f"[NAS-ATM4] state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm4/scan', methods=['POST'])
def api_nas_atm4_scan():
    """Trigger a manual NAS ATM4 scan (entry attempt)."""
    task_id = f"nas_atm4_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _nas_atm4_tasks[task_id] = {'status': 'running'}

    def _run_scan(tid):
        try:
            from services.nas_atm4_executor import NasAtm4Executor
            executor = NasAtm4Executor(config=NAS_ATM4_DEFAULTS)
            sid, msg = executor.execute_strangle_entry()
            _nas_atm4_tasks[tid] = {'status': 'completed', 'result': {'strangle_id': sid, 'message': msg}}
        except Exception as e:
            logger.error(f"[NAS-ATM4] scan error: {e}")
            _nas_atm4_tasks[tid] = {'status': 'error', 'error': str(e)}

    scheduler.add_job(_run_scan, args=[task_id], id=f'nas_atm4_scan_{task_id}',
                      replace_existing=True)
    return jsonify({'task_id': task_id, 'status': 'queued'})


@app.route('/api/nas-atm4/scan/status/<task_id>')
def api_nas_atm4_scan_status(task_id):
    """Poll NAS ATM4 scan status."""
    task = _nas_atm4_tasks.get(task_id, {'status': 'unknown'})
    return jsonify(task)


@app.route('/api/nas-atm4/kill-switch', methods=['POST'])
def api_nas_atm4_kill_switch():
    """Emergency close all NAS ATM4 positions."""
    try:
        from services.nas_atm4_executor import NasAtm4Executor
        executor = NasAtm4Executor(config=NAS_ATM4_DEFAULTS)
        exits = executor.emergency_exit_all()
        return jsonify({'status': 'killed', 'positions_closed': exits})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm4/trades')
def api_nas_atm4_trades():
    """Recent NAS ATM4 trades."""
    try:
        from services.nas_atm4_db import get_nas_atm4_db
        db = get_nas_atm4_db()
        return jsonify(db.get_recent_trades(limit=50))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm4/signals')
def api_nas_atm4_signals():
    """NAS ATM4 signal history."""
    try:
        from services.nas_atm4_db import get_nas_atm4_db
        db = get_nas_atm4_db()
        return jsonify(db.get_recent_signals(limit=30))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm4/equity-curve')
def api_nas_atm4_equity_curve():
    """NAS ATM4 daily P&L for equity curve chart."""
    try:
        from services.nas_atm4_db import get_nas_atm4_db
        db = get_nas_atm4_db()
        return jsonify(db.get_equity_curve())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm4/toggle-mode', methods=['POST'])
def api_nas_atm4_toggle_mode():
    """Toggle NAS ATM4 paper/live trading mode."""
    try:
        current = NAS_ATM4_DEFAULTS.get('paper_trading_mode', True)
        NAS_ATM4_DEFAULTS['paper_trading_mode'] = not current
        new_mode = 'PAPER' if NAS_ATM4_DEFAULTS['paper_trading_mode'] else 'LIVE'
        logger.info(f"[NAS-ATM4] mode toggled to {new_mode}")
        return jsonify({'mode': new_mode})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm4/toggle-enabled', methods=['POST'])
def api_nas_atm4_toggle_enabled():
    """Enable/disable NAS ATM4 system."""
    try:
        current = NAS_ATM4_DEFAULTS.get('enabled', True)
        NAS_ATM4_DEFAULTS['enabled'] = not current
        new_enabled = NAS_ATM4_DEFAULTS['enabled']
        status = 'enabled' if new_enabled else 'disabled'
        logger.info(f"[NAS-ATM4] system {status}")
        return jsonify({'enabled': new_enabled, 'status': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm4/ticker/start', methods=['POST'])
def api_nas_atm4_ticker_start():
    """Start NAS ticker (shared with OTM/ATM) for ATM4 system."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        if not ticker.is_running:
            ticker.start()
        return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm4/ticker/stop', methods=['POST'])
def api_nas_atm4_ticker_stop():
    """Stop NAS ticker (shared)."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker()
        ticker.stop()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm4/ticker/status')
def api_nas_atm4_ticker_status():
    """Get NAS ticker status (shared)."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        return jsonify(ticker.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-atm4/ticker/stream')
def api_nas_atm4_ticker_stream():
    """SSE stream for NAS ATM4 tick-by-tick updates."""
    import time as _time

    def generate():
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        last_spot = 0
        while True:
            _time.sleep(1)
            if not ticker.is_running:
                yield f"data: {json.dumps({'type': 'status', 'running': False})}\n\n"
                continue
            spot = ticker._last_ltp
            if spot != last_spot and spot > 0:
                last_spot = spot
                atm4_legs = {}
                for token, info in ticker._atm4_option_tokens.items():
                    tsym = info['tradingsymbol']
                    ltp = ticker._atm4_option_ltps.get(token)
                    if ltp is not None:
                        atm4_legs[tsym] = {'ltp': ltp, 'sl': info.get('sl_price', 0)}
                yield f"data: {json.dumps({'type': 'tick', 'spot': round(spot, 2), 'legs': atm4_legs})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ---- NAS ATM4 Scheduled Jobs ----

def _nas_atm4_eod_squareoff():
    """3:15 PM — EOD squareoff for NAS ATM4."""
    if not NAS_ATM4_DEFAULTS.get('enabled', True):
        return
    try:
        from services.nas_atm4_executor import NasAtm4Executor
        executor = NasAtm4Executor(config=NAS_ATM4_DEFAULTS)
        exits = executor.eod_squareoff()
        logger.info(f"[NAS-ATM4] EOD squareoff: {len(exits)} positions closed")

        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker()
        ticker.subscribe_atm4_option_legs([])
    except Exception as e:
        logger.error(f"[NAS-ATM4] EOD squareoff error: {e}")


def _nas_atm4_daily_summary():
    """3:20 PM — Daily summary for NAS ATM4."""
    if not NAS_ATM4_DEFAULTS.get('enabled', True):
        return
    try:
        from services.nas_atm4_db import get_nas_atm4_db
        db = get_nas_atm4_db()
        stats = db.get_stats()
        logger.info(f"[NAS-ATM4] Daily summary: {stats.get('total_trades', 0)} total trades, "
                     f"P&L: Rs {stats.get('total_pnl', 0):.0f}")
    except Exception as e:
        logger.error(f"[NAS-ATM4] Daily summary error: {e}")


try:
    scheduler.add_job(
        _nas_atm4_eod_squareoff,
        'cron', hour=15, minute=15, day_of_week='mon-fri',
        id='nas_atm4_eod_squareoff', replace_existing=True,
    )
    scheduler.add_job(
        _nas_atm4_daily_summary,
        'cron', hour=15, minute=20, day_of_week='mon-fri',
        id='nas_atm4_daily_summary', replace_existing=True,
    )
    logger.info(
        "NAS ATM4 scheduled jobs registered: EOD squareoff(15:15), daily summary(15:20)"
    )
except Exception as e:
    logger.warning(f"Could not register NAS ATM4 scheduled jobs: {e}")


# =============================================================================
# NAS 916 — All 4 strategies with mandatory 9:16 AM entry (no squeeze wait)
# =============================================================================

_nas_916_tasks = {}  # Shared task dict for all 916 scan polling


# ---- 916 OTM ----

@app.route('/api/nas-916-otm/state')
def api_nas_916_otm_state():
    try:
        from services.nas_916_executors import Nas916OtmExecutor
        executor = Nas916OtmExecutor(config=NAS_916_OTM_DEFAULTS)
        state = executor.get_full_state()
        state = _enrich_nas_positions_with_ltp(state, '_option_ltps', '_option_tokens')
        return jsonify(state)
    except Exception as e:
        logger.error(f"[NAS-916-OTM] state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-otm/scan', methods=['POST'])
def api_nas_916_otm_scan():
    task_id = f"nas_916_otm_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _nas_916_tasks[task_id] = {'status': 'running'}

    def _run(tid):
        try:
            from services.nas_916_executors import Nas916OtmExecutor
            executor = Nas916OtmExecutor(config=NAS_916_OTM_DEFAULTS)
            result = executor.run_scan()
            entries = result.get('entries', [])
            sid = entries[0]['strangle_id'] if entries else None
            _nas_916_tasks[tid] = {'status': 'completed', 'result': {'strangle_id': sid, 'message': 'OK' if sid else 'No entry'}}
        except Exception as e:
            logger.error(f"[NAS-916-OTM] scan error: {e}")
            _nas_916_tasks[tid] = {'status': 'error', 'error': str(e)}

    scheduler.add_job(_run, args=[task_id], id=f'nas_916_otm_scan_{task_id}', replace_existing=True)
    return jsonify({'task_id': task_id, 'status': 'queued'})


@app.route('/api/nas-916-otm/scan/status/<task_id>')
def api_nas_916_otm_scan_status(task_id):
    return jsonify(_nas_916_tasks.get(task_id, {'status': 'unknown'}))


@app.route('/api/nas-916-otm/kill-switch', methods=['POST'])
def api_nas_916_otm_kill():
    try:
        from services.nas_916_executors import Nas916OtmExecutor
        executor = Nas916OtmExecutor(config=NAS_916_OTM_DEFAULTS)
        exits = executor.exit_all_positions('KILL_SWITCH', {})
        return jsonify({'status': 'killed', 'positions_closed': len(exits)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-otm/trades')
def api_nas_916_otm_trades():
    try:
        from services.nas_916_db import get_nas_916_otm_db
        db = get_nas_916_otm_db()
        return jsonify(db.get_recent_trades(limit=50))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-otm/equity-curve')
def api_nas_916_otm_equity_curve():
    try:
        from services.nas_916_db import get_nas_916_otm_db
        db = get_nas_916_otm_db()
        return jsonify(db.get_equity_curve())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-otm/toggle-mode', methods=['POST'])
def api_nas_916_otm_toggle_mode():
    try:
        current = NAS_916_OTM_DEFAULTS.get('paper_trading_mode', True)
        NAS_916_OTM_DEFAULTS['paper_trading_mode'] = not current
        return jsonify({'mode': 'PAPER' if NAS_916_OTM_DEFAULTS['paper_trading_mode'] else 'LIVE'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-otm/toggle-enabled', methods=['POST'])
def api_nas_916_otm_toggle_enabled():
    try:
        current = NAS_916_OTM_DEFAULTS.get('enabled', True)
        NAS_916_OTM_DEFAULTS['enabled'] = not current
        return jsonify({'enabled': NAS_916_OTM_DEFAULTS['enabled']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-otm/ticker/stream')
def api_nas_916_otm_ticker_stream():
    """SSE stream — reuses OTM ticker tokens (same strategy, different DB)."""
    import time as _time

    def generate():
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        last_spot = 0
        while True:
            _time.sleep(1)
            if not ticker.is_running:
                yield f"data: {json.dumps({'type': 'status', 'running': False})}\n\n"
                continue
            spot = ticker._last_ltp
            if spot != last_spot and spot > 0:
                last_spot = spot
                legs = {}
                for token, info in ticker._option_tokens.items():
                    tsym = info['tradingsymbol']
                    ltp = ticker._option_ltps.get(token)
                    if ltp is not None:
                        legs[tsym] = {'ltp': ltp, 'sl': info.get('sl_price', 0)}
                yield f"data: {json.dumps({'type': 'tick', 'spot': round(spot, 2), 'legs': legs})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ---- 916 ATM ----

@app.route('/api/nas-916-atm/state')
def api_nas_916_atm_state():
    try:
        from services.nas_916_executors import Nas916AtmExecutor
        executor = Nas916AtmExecutor(config=NAS_916_ATM_DEFAULTS)
        state = executor.get_full_state()
        state = _enrich_nas_positions_with_ltp(state, '_atm_option_ltps', '_atm_option_tokens')
        return jsonify(state)
    except Exception as e:
        logger.error(f"[NAS-916-ATM] state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm/scan', methods=['POST'])
def api_nas_916_atm_scan():
    task_id = f"nas_916_atm_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _nas_916_tasks[task_id] = {'status': 'running'}

    def _run(tid):
        try:
            from services.nas_916_executors import Nas916AtmExecutor
            executor = Nas916AtmExecutor(config=NAS_916_ATM_DEFAULTS)
            sid, msg = executor.execute_strangle_entry()
            _nas_916_tasks[tid] = {'status': 'completed', 'result': {'strangle_id': sid, 'message': msg}}
        except Exception as e:
            logger.error(f"[NAS-916-ATM] scan error: {e}")
            _nas_916_tasks[tid] = {'status': 'error', 'error': str(e)}

    scheduler.add_job(_run, args=[task_id], id=f'nas_916_atm_scan_{task_id}', replace_existing=True)
    return jsonify({'task_id': task_id, 'status': 'queued'})


@app.route('/api/nas-916-atm/scan/status/<task_id>')
def api_nas_916_atm_scan_status(task_id):
    return jsonify(_nas_916_tasks.get(task_id, {'status': 'unknown'}))


@app.route('/api/nas-916-atm/kill-switch', methods=['POST'])
def api_nas_916_atm_kill():
    try:
        from services.nas_916_executors import Nas916AtmExecutor
        executor = Nas916AtmExecutor(config=NAS_916_ATM_DEFAULTS)
        exits = executor.emergency_exit_all()
        return jsonify({'status': 'killed', 'positions_closed': exits})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm/trades')
def api_nas_916_atm_trades():
    try:
        from services.nas_916_db import get_nas_916_atm_db
        db = get_nas_916_atm_db()
        return jsonify(db.get_recent_trades(limit=50))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm/equity-curve')
def api_nas_916_atm_equity_curve():
    try:
        from services.nas_916_db import get_nas_916_atm_db
        db = get_nas_916_atm_db()
        return jsonify(db.get_equity_curve())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm/toggle-mode', methods=['POST'])
def api_nas_916_atm_toggle_mode():
    try:
        current = NAS_916_ATM_DEFAULTS.get('paper_trading_mode', True)
        NAS_916_ATM_DEFAULTS['paper_trading_mode'] = not current
        return jsonify({'mode': 'PAPER' if NAS_916_ATM_DEFAULTS['paper_trading_mode'] else 'LIVE'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm/toggle-enabled', methods=['POST'])
def api_nas_916_atm_toggle_enabled():
    try:
        current = NAS_916_ATM_DEFAULTS.get('enabled', True)
        NAS_916_ATM_DEFAULTS['enabled'] = not current
        return jsonify({'enabled': NAS_916_ATM_DEFAULTS['enabled']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm/ticker/stream')
def api_nas_916_atm_ticker_stream():
    import time as _time

    def generate():
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        last_spot = 0
        while True:
            _time.sleep(1)
            if not ticker.is_running:
                yield f"data: {json.dumps({'type': 'status', 'running': False})}\n\n"
                continue
            spot = ticker._last_ltp
            if spot != last_spot and spot > 0:
                last_spot = spot
                legs = {}
                for token, info in ticker._atm_option_tokens.items():
                    tsym = info['tradingsymbol']
                    ltp = ticker._atm_option_ltps.get(token)
                    if ltp is not None:
                        legs[tsym] = {'ltp': ltp, 'sl': info.get('sl_price', 0)}
                yield f"data: {json.dumps({'type': 'tick', 'spot': round(spot, 2), 'legs': legs})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ---- 916 ATM2 ----

@app.route('/api/nas-916-atm2/state')
def api_nas_916_atm2_state():
    try:
        from services.nas_916_executors import Nas916Atm2Executor
        executor = Nas916Atm2Executor(config=NAS_916_ATM2_DEFAULTS)
        state = executor.get_full_state()
        state = _enrich_nas_positions_with_ltp(state, '_atm2_option_ltps', '_atm2_option_tokens')
        return jsonify(state)
    except Exception as e:
        logger.error(f"[NAS-916-ATM2] state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm2/scan', methods=['POST'])
def api_nas_916_atm2_scan():
    task_id = f"nas_916_atm2_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _nas_916_tasks[task_id] = {'status': 'running'}

    def _run(tid):
        try:
            from services.nas_916_executors import Nas916Atm2Executor
            executor = Nas916Atm2Executor(config=NAS_916_ATM2_DEFAULTS)
            sid, msg = executor.execute_strangle_entry()
            _nas_916_tasks[tid] = {'status': 'completed', 'result': {'strangle_id': sid, 'message': msg}}
        except Exception as e:
            logger.error(f"[NAS-916-ATM2] scan error: {e}")
            _nas_916_tasks[tid] = {'status': 'error', 'error': str(e)}

    scheduler.add_job(_run, args=[task_id], id=f'nas_916_atm2_scan_{task_id}', replace_existing=True)
    return jsonify({'task_id': task_id, 'status': 'queued'})


@app.route('/api/nas-916-atm2/scan/status/<task_id>')
def api_nas_916_atm2_scan_status(task_id):
    return jsonify(_nas_916_tasks.get(task_id, {'status': 'unknown'}))


@app.route('/api/nas-916-atm2/kill-switch', methods=['POST'])
def api_nas_916_atm2_kill():
    try:
        from services.nas_916_executors import Nas916Atm2Executor
        executor = Nas916Atm2Executor(config=NAS_916_ATM2_DEFAULTS)
        exits = executor.emergency_exit_all()
        return jsonify({'status': 'killed', 'positions_closed': exits})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm2/trades')
def api_nas_916_atm2_trades():
    try:
        from services.nas_916_db import get_nas_916_atm2_db
        db = get_nas_916_atm2_db()
        return jsonify(db.get_recent_trades(limit=50))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm2/equity-curve')
def api_nas_916_atm2_equity_curve():
    try:
        from services.nas_916_db import get_nas_916_atm2_db
        db = get_nas_916_atm2_db()
        return jsonify(db.get_equity_curve())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm2/toggle-mode', methods=['POST'])
def api_nas_916_atm2_toggle_mode():
    try:
        current = NAS_916_ATM2_DEFAULTS.get('paper_trading_mode', True)
        NAS_916_ATM2_DEFAULTS['paper_trading_mode'] = not current
        return jsonify({'mode': 'PAPER' if NAS_916_ATM2_DEFAULTS['paper_trading_mode'] else 'LIVE'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm2/toggle-enabled', methods=['POST'])
def api_nas_916_atm2_toggle_enabled():
    try:
        current = NAS_916_ATM2_DEFAULTS.get('enabled', True)
        NAS_916_ATM2_DEFAULTS['enabled'] = not current
        return jsonify({'enabled': NAS_916_ATM2_DEFAULTS['enabled']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm2/ticker/stream')
def api_nas_916_atm2_ticker_stream():
    import time as _time

    def generate():
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        last_spot = 0
        while True:
            _time.sleep(1)
            if not ticker.is_running:
                yield f"data: {json.dumps({'type': 'status', 'running': False})}\n\n"
                continue
            spot = ticker._last_ltp
            if spot != last_spot and spot > 0:
                last_spot = spot
                legs = {}
                for token, info in ticker._atm2_option_tokens.items():
                    tsym = info['tradingsymbol']
                    ltp = ticker._atm2_option_ltps.get(token)
                    if ltp is not None:
                        legs[tsym] = {'ltp': ltp, 'sl': info.get('sl_price', 0)}
                yield f"data: {json.dumps({'type': 'tick', 'spot': round(spot, 2), 'legs': legs})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ---- 916 ATM4 ----

@app.route('/api/nas-916-atm4/state')
def api_nas_916_atm4_state():
    try:
        from services.nas_916_executors import Nas916Atm4Executor
        executor = Nas916Atm4Executor(config=NAS_916_ATM4_DEFAULTS)
        state = executor.get_full_state()
        state = _enrich_nas_positions_with_ltp(state, '_atm4_option_ltps', '_atm4_option_tokens')
        return jsonify(state)
    except Exception as e:
        logger.error(f"[NAS-916-ATM4] state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm4/scan', methods=['POST'])
def api_nas_916_atm4_scan():
    task_id = f"nas_916_atm4_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _nas_916_tasks[task_id] = {'status': 'running'}

    def _run(tid):
        try:
            from services.nas_916_executors import Nas916Atm4Executor
            executor = Nas916Atm4Executor(config=NAS_916_ATM4_DEFAULTS)
            sid, msg = executor.execute_strangle_entry()
            _nas_916_tasks[tid] = {'status': 'completed', 'result': {'strangle_id': sid, 'message': msg}}
        except Exception as e:
            logger.error(f"[NAS-916-ATM4] scan error: {e}")
            _nas_916_tasks[tid] = {'status': 'error', 'error': str(e)}

    scheduler.add_job(_run, args=[task_id], id=f'nas_916_atm4_scan_{task_id}', replace_existing=True)
    return jsonify({'task_id': task_id, 'status': 'queued'})


@app.route('/api/nas-916-atm4/scan/status/<task_id>')
def api_nas_916_atm4_scan_status(task_id):
    return jsonify(_nas_916_tasks.get(task_id, {'status': 'unknown'}))


@app.route('/api/nas-916-atm4/kill-switch', methods=['POST'])
def api_nas_916_atm4_kill():
    try:
        from services.nas_916_executors import Nas916Atm4Executor
        executor = Nas916Atm4Executor(config=NAS_916_ATM4_DEFAULTS)
        exits = executor.emergency_exit_all()
        return jsonify({'status': 'killed', 'positions_closed': exits})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm4/trades')
def api_nas_916_atm4_trades():
    try:
        from services.nas_916_db import get_nas_916_atm4_db
        db = get_nas_916_atm4_db()
        return jsonify(db.get_recent_trades(limit=50))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm4/equity-curve')
def api_nas_916_atm4_equity_curve():
    try:
        from services.nas_916_db import get_nas_916_atm4_db
        db = get_nas_916_atm4_db()
        return jsonify(db.get_equity_curve())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm4/toggle-mode', methods=['POST'])
def api_nas_916_atm4_toggle_mode():
    try:
        current = NAS_916_ATM4_DEFAULTS.get('paper_trading_mode', True)
        NAS_916_ATM4_DEFAULTS['paper_trading_mode'] = not current
        return jsonify({'mode': 'PAPER' if NAS_916_ATM4_DEFAULTS['paper_trading_mode'] else 'LIVE'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm4/toggle-enabled', methods=['POST'])
def api_nas_916_atm4_toggle_enabled():
    try:
        current = NAS_916_ATM4_DEFAULTS.get('enabled', True)
        NAS_916_ATM4_DEFAULTS['enabled'] = not current
        return jsonify({'enabled': NAS_916_ATM4_DEFAULTS['enabled']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas-916-atm4/ticker/stream')
def api_nas_916_atm4_ticker_stream():
    import time as _time

    def generate():
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker(NAS_DEFAULTS)
        last_spot = 0
        while True:
            _time.sleep(1)
            if not ticker.is_running:
                yield f"data: {json.dumps({'type': 'status', 'running': False})}\n\n"
                continue
            spot = ticker._last_ltp
            if spot != last_spot and spot > 0:
                last_spot = spot
                legs = {}
                for token, info in ticker._atm4_option_tokens.items():
                    tsym = info['tradingsymbol']
                    ltp = ticker._atm4_option_ltps.get(token)
                    if ltp is not None:
                        legs[tsym] = {'ltp': ltp, 'sl': info.get('sl_price', 0)}
                yield f"data: {json.dumps({'type': 'tick', 'spot': round(spot, 2), 'legs': legs})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ---- NAS 916 Scheduled Jobs ----

def _nas_916_auto_entry():
    """9:16 AM Mon-Fri — Auto-enter all 4 x 916 systems.

    Per-system Wed/Thu filter respected via cfg['skip_weekdays']:
      - 916 OTM: inherits NAS_DEFAULTS.skip_weekdays = (2, 3) → Wed/Thu OFF
      - 916 ATM: no skip_weekdays → enabled all weekdays
      - 916 ATM2 / 916 ATM4: inherit (2, 3) → Wed/Thu OFF
    """
    today_wd = datetime.now().weekday()
    systems = [
        ('NAS-916-OTM', NAS_916_OTM_DEFAULTS, 'services.nas_916_executors', 'Nas916OtmExecutor', 'run_scan'),
        ('NAS-916-ATM', NAS_916_ATM_DEFAULTS, 'services.nas_916_executors', 'Nas916AtmExecutor', 'execute_strangle_entry'),
        ('NAS-916-ATM2', NAS_916_ATM2_DEFAULTS, 'services.nas_916_executors', 'Nas916Atm2Executor', 'execute_strangle_entry'),
        ('NAS-916-ATM4', NAS_916_ATM4_DEFAULTS, 'services.nas_916_executors', 'Nas916Atm4Executor', 'execute_strangle_entry'),
    ]
    for name, cfg, mod_path, cls_name, method in systems:
        if not cfg.get('enabled', True):
            logger.info(f"[{name}] disabled, skipping auto-entry")
            continue
        skip_days = cfg.get('skip_weekdays') or ()
        if today_wd in skip_days:
            logger.info(f"[{name}] weekday {today_wd} in skip_weekdays={skip_days}, no auto-entry today")
            continue
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            executor = cls(config=cfg)
            if method == 'run_scan':
                result = executor.run_scan()
                entries = result.get('entries', [])
                logger.info(f"[{name}] 9:16 auto-entry: {len(entries)} entries")
            else:
                sid, msg = getattr(executor, method)()
                logger.info(f"[{name}] 9:16 auto-entry: sid={sid}, {msg}")
        except Exception as e:
            logger.error(f"[{name}] 9:16 auto-entry error: {e}")


def _nas_916_eod_squareoff():
    """3:15 PM — EOD squareoff for all 916 systems."""
    systems = [
        ('NAS-916-OTM', NAS_916_OTM_DEFAULTS, 'services.nas_916_executors', 'Nas916OtmExecutor', 'exit_all_positions'),
        ('NAS-916-ATM', NAS_916_ATM_DEFAULTS, 'services.nas_916_executors', 'Nas916AtmExecutor', 'eod_squareoff'),
        ('NAS-916-ATM2', NAS_916_ATM2_DEFAULTS, 'services.nas_916_executors', 'Nas916Atm2Executor', 'eod_squareoff'),
        ('NAS-916-ATM4', NAS_916_ATM4_DEFAULTS, 'services.nas_916_executors', 'Nas916Atm4Executor', 'eod_squareoff'),
    ]
    for name, cfg, mod_path, cls_name, method in systems:
        if not cfg.get('enabled', True):
            continue
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            executor = cls(config=cfg)
            if method == 'exit_all_positions':
                exits = executor.exit_all_positions('EOD_SQUAREOFF', {})
                logger.info(f"[{name}] EOD squareoff: {len(exits)} positions closed")
            else:
                exits = executor.eod_squareoff()
                logger.info(f"[{name}] EOD squareoff: {len(exits)} positions closed")
        except Exception as e:
            logger.error(f"[{name}] EOD squareoff error: {e}")


def _nas_916_sl_monitor():
    """Poll each 9:16 system's active positions every 10s and run its own
    check_and_handle_sl(). Fixes the gap where nas_ticker.py only wires SL
    handling for Squeeze variants — 9:16 positions had no SL enforcement
    until this job was added (discovered 2026-04-21: 9:16 ATM V4 CE went
    from entry 59 to 122.55 without SL at 76.70 firing)."""
    variants = [
        ('NAS-916-OTM',  'NAS_916_OTM_DEFAULTS',  'Nas916OtmExecutor'),
        ('NAS-916-ATM',  'NAS_916_ATM_DEFAULTS',  'Nas916AtmExecutor'),
        ('NAS-916-ATM2', 'NAS_916_ATM2_DEFAULTS', 'Nas916Atm2Executor'),
        ('NAS-916-ATM4', 'NAS_916_ATM4_DEFAULTS', 'Nas916Atm4Executor'),
    ]
    # Only run during market hours (9:15-15:30 IST)
    now = datetime.now()
    if not (now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 30)):
        return

    from services.nas_916_executors import (
        Nas916OtmExecutor, Nas916AtmExecutor,
        Nas916Atm2Executor, Nas916Atm4Executor,
    )
    from services.kite_service import get_kite

    name_to_cls = {
        'Nas916OtmExecutor': Nas916OtmExecutor,
        'Nas916AtmExecutor': Nas916AtmExecutor,
        'Nas916Atm2Executor': Nas916Atm2Executor,
        'Nas916Atm4Executor': Nas916Atm4Executor,
    }
    name_to_cfg = {
        'NAS_916_OTM_DEFAULTS': NAS_916_OTM_DEFAULTS,
        'NAS_916_ATM_DEFAULTS': NAS_916_ATM_DEFAULTS,
        'NAS_916_ATM2_DEFAULTS': NAS_916_ATM2_DEFAULTS,
        'NAS_916_ATM4_DEFAULTS': NAS_916_ATM4_DEFAULTS,
    }

    try:
        kite = get_kite()
    except Exception as e:
        logger.warning(f"[NAS-916-SL] Kite not available: {e}")
        return

    for label, cfg_name, cls_name in variants:
        try:
            cfg = name_to_cfg[cfg_name]
            if not cfg.get('enabled', True):
                continue
            executor = name_to_cls[cls_name](config=cfg)
            active = executor.db.get_active_positions() or []
            if not active:
                continue
            # Batch LTP fetch for all active tradingsymbols
            syms = list({p['tradingsymbol'] for p in active if p.get('tradingsymbol')})
            if not syms:
                continue
            ltp_map = {}
            try:
                keys = [f'NFO:{s}' for s in syms]
                quote_resp = kite.ltp(keys) or {}
                for s in syms:
                    v = quote_resp.get(f'NFO:{s}')
                    if v and v.get('last_price'):
                        ltp_map[s] = v['last_price']
            except Exception as e:
                logger.warning(f"[{label}] ltp fetch failed: {e}")
                continue
            if not ltp_map:
                continue
            # OTM (NasExecutor base) has no check_and_handle_sl — OTM uses
            # cross-leg adjustment, not per-leg SL. Skip it silently.
            if not hasattr(executor, 'check_and_handle_sl'):
                continue
            actions = executor.check_and_handle_sl(positions=active, live_ltps=ltp_map)
            if actions:
                logger.info(f"[{label}] SL monitor: {len(actions)} actions")
        except Exception as e:
            logger.error(f"[{label}] SL monitor error: {e}", exc_info=True)


try:
    # Auto-entry at 9:16 AM (all weekdays — Thursday included for paper data)
    scheduler.add_job(
        _nas_916_auto_entry,
        'cron', day_of_week='mon-fri', hour=9, minute=16,
        id='nas_916_auto_entry', replace_existing=True,
    )
    # SL monitor — poll every 10s during market hours (fills the gap in nas_ticker)
    scheduler.add_job(
        _nas_916_sl_monitor,
        'interval', seconds=10,
        id='nas_916_sl_monitor', replace_existing=True,
    )
    # EOD squareoff at 3:15 PM
    scheduler.add_job(
        _nas_916_eod_squareoff,
        'cron', hour=15, minute=15, day_of_week='mon-fri',
        id='nas_916_eod_squareoff', replace_existing=True,
    )
    logger.info(
        "NAS 916 scheduled jobs registered: auto-entry(9:16), SL monitor(10s poll), "
        "EOD squareoff(15:15) — Mon-Fri"
    )
except Exception as e:
    logger.warning(f"Could not register NAS 916 scheduled jobs: {e}")


# =============================================================================
# ORB — Opening Range Breakout (Cash Equity Intraday)
# =============================================================================

_orb_tasks = {}


def _get_orb_db():
    """Lazy singleton for ORB database."""
    from services.orb_db import OrbDB
    if not hasattr(_get_orb_db, '_instance'):
        _get_orb_db._instance = OrbDB()
    return _get_orb_db._instance


@app.route('/orb')
def orb_dashboard():
    # Legacy Jinja page retired — redirect to React SPA at /app/orb.
    # Template templates/orb_dashboard.html kept in repo as backup; swap
    # this body back to render_template('orb_dashboard.html', ...) to restore.
    return redirect('/app/orb', code=302)


_orb_cache = {'margin': None, 'margin_ts': 0, 'ltps': {}, 'ltp_ts': 0}

def _orb_get_margin():
    """Return cached margin. Background refresh every 5 min.

    Field semantics (verified against Kite Funds page 2026-04-30):
      eq.net               = the headline 'Available margin' shown in Kite UI
                             (= cash + collateral - utilised.debits)
      eq.available.cash    = opening balance (start-of-day cash); does NOT
                             reflect today's debits — misleading for sizing
      eq.available.live_bal = instantaneous deployable cash for new MIS;
                             can go small/negative when FnO margin uses
                             collateral
      eq.utilised.debits   = total margin currently blocked

    We expose 'available' = net so the dashboard tile matches what the user
    sees in Kite's Funds page. 'cash' / 'live_balance' / 'used' kept for
    detail-level callers.
    """
    import time as _t, threading
    now = _t.time()
    if now - _orb_cache['margin_ts'] > 300:
        _orb_cache['margin_ts'] = now  # prevent parallel fetches
        def _bg():
            try:
                kite = get_kite()
                margins = kite.margins()
                eq = margins.get('equity', {})
                avail = eq.get('available', {})
                used = eq.get('utilised', {}).get('debits', 0)
                net = eq.get('net', 0)
                cash = avail.get('cash', 0)
                live_balance = avail.get('live_balance', 0)
                _orb_cache['margin'] = {
                    'available': round(net, 2),  # matches Kite UI 'Available margin'
                    'cash': round(cash, 2),
                    'live_balance': round(live_balance, 2),
                    'used': round(used, 2),
                }
            except Exception:
                pass
        threading.Thread(target=_bg, daemon=True).start()
    return _orb_cache['margin']


def _orb_get_ltps(symbols):
    """Return cached LTPs. Background refresh every 10 sec."""
    import time as _t, threading
    now = _t.time()
    if now - _orb_cache['ltp_ts'] > 10:
        _orb_cache['ltp_ts'] = now
        def _bg():
            try:
                kite = get_kite()
                ltps = kite.ltp(['NSE:' + s for s in symbols])
                for s in symbols:
                    ltp = ltps.get('NSE:' + s, {}).get('last_price')
                    if ltp:
                        _orb_cache['ltps'][s] = round(ltp, 2)
            except Exception:
                pass
        threading.Thread(target=_bg, daemon=True).start()
    return _orb_cache['ltps']


def _orb_check_fund_alert():
    """Check if available margin is below 1.2x per-trade allocation."""
    margin = _orb_get_margin()
    if not margin:
        return None
    capital = ORB_DEFAULTS.get('capital', 100000)
    max_trades = ORB_DEFAULTS.get('max_concurrent_trades', 3)
    buffer = ORB_DEFAULTS.get('margin_buffer_multiplier', 1.2)
    alloc = capital / max_trades
    min_required = alloc * buffer
    available = margin.get('available', 0)
    if available < min_required:
        return {
            'type': 'warning',
            'message': f'Low funds: Rs {available:,.0f} available, need Rs {min_required:,.0f} (1.2x of Rs {alloc:,.0f} per-trade alloc)',
            'available': available,
            'required': min_required,
        }
    return None


@app.route('/api/orb/state')
def api_orb_state():
    """Full state dump for ORB dashboard: daily states, positions, stats."""
    try:
        db = _get_orb_db()
        state = db.get_state()
        stats = db.get_stats()
        today_closed = db.get_today_closed()

        # Build per-stock summary from daily states + positions
        stocks = {}
        alloc = round(
            ORB_DEFAULTS.get('capital', 100000)
            * ORB_DEFAULTS.get('mis_leverage', 1)
            / ORB_DEFAULTS.get('max_concurrent_trades', 3)
        )
        for ds in state.get('daily_states', []):
            sym = ds['instrument']
            today_open = ds.get('today_open') or 0
            qty = int(alloc // today_open) if today_open > 0 else 0
            capital_per_trade = round(qty * today_open) if today_open > 0 else 0
            or_high = ds.get('or_high') or 0
            or_low = ds.get('or_low') or 0
            sl_risk = round(abs(or_high - or_low) * qty) if or_high and or_low else 0
            stocks[sym] = {
                'daily_state': ds,
                'position': None,
                'today_result': None,
                'qty': qty,
                'capital_per_trade': capital_per_trade,
                'sl_risk_inr': sl_risk,
                'price': today_open,
            }

        # Enrich open positions with cached LTP (non-blocking) + parse conviction_stars JSON
        open_positions = state.get('open_positions', [])
        if open_positions:
            ltp_syms = [p['instrument'] for p in open_positions]
            ltps = _orb_get_ltps(ltp_syms)
            import json as _json_pos
            for pos in open_positions:
                ltp = ltps.get(pos['instrument'])
                if ltp:
                    pos['ltp'] = ltp
                    entry = pos['entry_price']
                    qty = pos['qty']
                    pos['pnl_pts'] = round((ltp - entry) if pos['direction'] == 'LONG' else (entry - ltp), 2)
                    pos['pnl_inr'] = round(pos['pnl_pts'] * qty, 2)
                stars = pos.get('conviction_stars')
                if isinstance(stars, str) and stars:
                    try:
                        pos['conviction_stars'] = _json_pos.loads(stars)
                    except Exception:
                        pass

        for pos in open_positions:
            sym = pos['instrument']
            if sym in stocks:
                stocks[sym]['position'] = pos

        # Parse conviction_stars JSON on closed trades too (frontend unified
        # Positions table iterates conviction_stars on every row).
        import json as _json_closed
        for pos in today_closed:
            stars = pos.get('conviction_stars')
            if isinstance(stars, str) and stars:
                try:
                    pos['conviction_stars'] = _json_closed.loads(stars)
                except Exception:
                    pos['conviction_stars'] = []
            sym = pos['instrument']
            if sym in stocks:
                stocks[sym]['today_result'] = pos

        # Today's P&L from open + closed positions
        closed_pnl = sum(p.get('pnl_inr', 0) or 0 for p in today_closed)
        open_pnl = sum(p.get('pnl_inr', 0) or 0 for p in open_positions if p.get('pnl_inr'))
        today_pnl = closed_pnl + open_pnl

        # Resolve daily loss limit (pct-of-capital unless override is set)
        _cap = ORB_DEFAULTS.get('capital', 100_000)
        _dll_override = ORB_DEFAULTS.get('daily_loss_limit')
        if _dll_override:
            _dll = float(_dll_override)
        else:
            _dll = round(_cap * float(ORB_DEFAULTS.get('daily_loss_limit_pct', 0.03)), 2)

        return jsonify({
            'enabled': ORB_DEFAULTS.get('enabled', True),
            'live_trading': ORB_DEFAULTS.get('live_trading_enabled', False),
            'universe': ORB_DEFAULTS.get('universe', []),
            'watch_universe': ORB_DEFAULTS.get('watch_universe', []),
            'capital': ORB_DEFAULTS.get('capital', 100_000),
            'mis_leverage': ORB_DEFAULTS.get('mis_leverage', 1),
            'use_risk_based_sizing': ORB_DEFAULTS.get('use_risk_based_sizing', False),
            'risk_per_trade_pct': ORB_DEFAULTS.get('risk_per_trade_pct', 0.0),
            'max_notional_per_trade': ORB_DEFAULTS.get('max_notional_per_trade', 0),
            'daily_loss_limit': _dll,
            'daily_loss_limit_pct': ORB_DEFAULTS.get('daily_loss_limit_pct', 0.03),
            'enforce_daily_loss_cap': ORB_DEFAULTS.get('enforce_daily_loss_cap', True),
            'stocks': stocks,
            'open_positions': state.get('open_positions', []),
            'today_closed': today_closed,
            'today_pnl': round(today_pnl, 2),
            'stats': stats,
            'margin': _orb_get_margin(),
            'fund_alert': _orb_check_fund_alert(),
            'unread_notifications': db.get_unread_count(),
            'config': {
                'capital': ORB_DEFAULTS.get('capital', 100000),
                'max_concurrent_trades': ORB_DEFAULTS.get('max_concurrent_trades', 3),
                'allocation_per_trade': round(
                    ORB_DEFAULTS.get('capital', 100000)
                    * ORB_DEFAULTS.get('mis_leverage', 1)
                    / ORB_DEFAULTS.get('max_concurrent_trades', 3)
                ),
                'min_margin_for_trade': round(ORB_DEFAULTS.get('capital', 100000) / ORB_DEFAULTS.get('max_concurrent_trades', 3) * ORB_DEFAULTS.get('margin_buffer_multiplier', 1.2)),
                'margin_buffer': ORB_DEFAULTS.get('margin_buffer_multiplier', 1.2),
                'or_minutes': ORB_DEFAULTS.get('or_minutes', 15),
                'r_multiple': ORB_DEFAULTS.get('r_multiple', 1.5),
                'sl_type': ORB_DEFAULTS.get('sl_type', 'or_opposite'),
                'last_entry_time': ORB_DEFAULTS.get('last_entry_time', '14:00'),
                'eod_exit_time': ORB_DEFAULTS.get('eod_exit_time', '15:20'),
                'use_vwap_filter': ORB_DEFAULTS.get('use_vwap_filter', True),
                'use_rsi_filter': ORB_DEFAULTS.get('use_rsi_filter', True),
                'use_cpr_dir_filter': ORB_DEFAULTS.get('use_cpr_dir_filter', True),
                'use_cpr_width_filter': ORB_DEFAULTS.get('use_cpr_width_filter', True),
                'cpr_width_threshold_pct': ORB_DEFAULTS.get('cpr_width_threshold_pct', 0.5),
                'use_gap_filter': ORB_DEFAULTS.get('use_gap_filter', True),
                'gap_threshold_pct': ORB_DEFAULTS.get('gap_threshold_pct', 0.3),
            },
        })
    except Exception as e:
        logger.error(f"ORB state error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/initialize', methods=['POST'])
def api_orb_initialize():
    """Manually trigger day initialization — uses Kite API for prev day HLC + CPR.
    Uses the shared singleton so state is visible to scheduled update_or/evaluate_signals."""
    try:
        from datetime import date as _date
        engine = _get_orb_engine()
        engine.initialize_day()

        # Return the computed state
        db = _get_orb_db()
        today = _date.today().isoformat()
        results = {}
        for sym in ORB_DEFAULTS.get('universe', []):
            ds = db.get_or_create_daily_state(sym, today)
            results[sym] = {
                'pivot': ds.get('cpr_pivot'),
                'tc': ds.get('cpr_tc'),
                'bc': ds.get('cpr_bc'),
                'cpr_width_pct': ds.get('cpr_width_pct'),
                'is_wide': bool(ds.get('is_wide_cpr_day')),
                'prev_day_date': ds.get('prev_day_date'),
                'gap_pct': ds.get('gap_pct'),
                'today_open': ds.get('today_open'),
            }

        return jsonify({'status': 'initialized', 'results': results})
    except Exception as e:
        logger.error(f"[ORB] Initialize error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/catchup', methods=['POST'])
def api_orb_catchup():
    """Recovery-only: take trades for breakouts missed earlier today if still valid.
    Walks today's 5-min candles, finds first post-OR breakout per stock, and
    places entries for stocks whose current LTP is still beyond OR. Respects
    all standard filters. Safer than auto — kept behind a manual POST."""
    try:
        engine = _get_orb_engine()
        result = engine.catchup_missed_breakouts()
        return jsonify(result)
    except Exception as e:
        logger.error(f"[ORB] catchup error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/candidates')
def api_orb_candidates():
    """Live candidate snapshot: broken_out, watching inside OR, excluded."""
    try:
        engine = _get_orb_engine()
        return jsonify(engine.get_candidates())
    except Exception as e:
        logger.error(f"[ORB] candidates error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/ensure-sl-orders', methods=['POST'])
def api_orb_ensure_sl_orders():
    """Place exchange SL-M orders for any OPEN position that lacks one.
    Also re-places if an existing SL-M was cancelled/rejected/invalid."""
    try:
        engine = _get_orb_engine()
        result = engine.ensure_sl_orders_placed()
        return jsonify(result)
    except Exception as e:
        logger.error(f"[ORB] ensure_sl_orders error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/update-or', methods=['POST'])
def api_orb_update_or():
    """Manually trigger OR-window update on the singleton engine.
    Useful for recovery after a mid-morning restart so or_high/or_low can be
    computed from already-completed 5-min candles (9:15, 9:20, 9:25)."""
    try:
        engine = _get_orb_engine()
        engine.update_or()
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"[ORB] manual update_or error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/scan', methods=['POST'])
def api_orb_scan():
    """Manually trigger signal evaluation across all stocks."""
    task_id = f"orb_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _orb_tasks[task_id] = {'status': 'running'}

    def _run_scan(tid):
        try:
            db = _get_orb_db()
            from datetime import date as _date
            today = _date.today()
            signals = []
            for sym in ORB_DEFAULTS.get('universe', []):
                ds = db.get_or_create_daily_state(sym, today)
                # Check if OR is finalized and no trade taken yet
                if ds.get('or_finalized') and ds.get('trades_taken', 0) < ORB_DEFAULTS.get('max_trades_per_day', 1):
                    signals.append({
                        'instrument': sym,
                        'or_high': ds.get('or_high'),
                        'or_low': ds.get('or_low'),
                        'cpr_width_pct': ds.get('cpr_width_pct'),
                        'is_wide_cpr_day': ds.get('is_wide_cpr_day'),
                        'status': 'ready_for_signal',
                    })
                else:
                    signals.append({
                        'instrument': sym,
                        'status': 'not_ready' if not ds.get('or_finalized') else 'trade_taken',
                    })
            _orb_tasks[tid] = {'status': 'completed', 'signals': signals}
        except Exception as e:
            logger.error(f"[ORB] Scan error: {e}")
            _orb_tasks[tid] = {'status': 'error', 'error': str(e)}

    scheduler.add_job(_run_scan, args=[task_id], id=f'orb_scan_{task_id}',
                      replace_existing=True)
    return jsonify({'task_id': task_id, 'status': 'queued'})


@app.route('/api/orb/scan/status/<task_id>')
def api_orb_scan_status(task_id):
    """Poll ORB scan status."""
    task = _orb_tasks.get(task_id, {'status': 'unknown'})
    return jsonify(task)


@app.route('/api/orb/kill-switch', methods=['POST'])
def api_orb_kill_switch():
    """Emergency close all open ORB positions."""
    try:
        db = _get_orb_db()
        open_positions = db.get_open_positions()
        closed = []

        if is_authenticated() and ORB_DEFAULTS.get('live_trading_enabled', False):
            # Live: place market sell orders via Kite
            try:
                dm = get_data_manager()
                kite = dm.kite
                for pos in open_positions:
                    tx_type = 'SELL' if pos['direction'] == 'LONG' else 'BUY'
                    try:
                        order_id = kite.place_order(
                            variety=kite.VARIETY_REGULAR,
                            exchange=kite.EXCHANGE_NSE,
                            tradingsymbol=pos['instrument'],
                            transaction_type=tx_type,
                            quantity=pos['qty'],
                            product=kite.PRODUCT_MIS,
                            order_type=kite.ORDER_TYPE_MARKET,
                        )
                        db.close_position(
                            pos['id'],
                            exit_price=pos.get('entry_price', 0),  # Approximate; real price from order
                            exit_time=datetime.now().isoformat(),
                            exit_reason='KILL_SWITCH',
                            pnl_pts=0,
                            pnl_inr=0,
                            kite_exit_order_id=str(order_id),
                        )
                        closed.append({'instrument': pos['instrument'], 'order_id': order_id})
                    except Exception as oe:
                        logger.error(f"[ORB] Kill switch order error {pos['instrument']}: {oe}")
                        closed.append({'instrument': pos['instrument'], 'error': str(oe)})
            except Exception as ke:
                logger.error(f"[ORB] Kill switch Kite error: {ke}")
                # Fall back to DB-only close
                for pos in open_positions:
                    db.close_position(
                        pos['id'],
                        exit_price=pos.get('entry_price', 0),
                        exit_time=datetime.now().isoformat(),
                        exit_reason='KILL_SWITCH',
                        pnl_pts=0, pnl_inr=0,
                    )
                    closed.append({'instrument': pos['instrument'], 'status': 'db_closed'})
        else:
            # DB-only close (no Kite)
            for pos in open_positions:
                db.close_position(
                    pos['id'],
                    exit_price=pos.get('entry_price', 0),
                    exit_time=datetime.now().isoformat(),
                    exit_reason='KILL_SWITCH',
                    pnl_pts=0, pnl_inr=0,
                )
                closed.append({'instrument': pos['instrument'], 'status': 'db_closed'})

        return jsonify({'closed': closed, 'count': len(closed), 'status': 'EMERGENCY_EXIT'})
    except Exception as e:
        logger.error(f"[ORB] Kill switch error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/toggle-enabled', methods=['POST'])
def api_orb_toggle_enabled():
    """Enable/disable ORB system."""
    try:
        current = ORB_DEFAULTS.get('enabled', True)
        ORB_DEFAULTS['enabled'] = not current
        new_status = 'ENABLED' if ORB_DEFAULTS['enabled'] else 'DISABLED'
        logger.info(f"[ORB] System {new_status}")
        return jsonify({'enabled': ORB_DEFAULTS['enabled'], 'status': new_status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/trades')
def api_orb_trades():
    """Recent ORB closed trades."""
    try:
        db = _get_orb_db()
        return jsonify(db.get_recent_trades(limit=50))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/signals')
def api_orb_signals():
    """ORB signal history."""
    try:
        db = _get_orb_db()
        return jsonify(db.get_recent_signals(limit=30))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/equity-curve')
def api_orb_equity_curve():
    """ORB daily P&L equity curve."""
    try:
        db = _get_orb_db()
        return jsonify(db.get_equity_curve())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/notifications')
def api_orb_notifications():
    """Get recent notifications for the notifications tab."""
    try:
        db = _get_orb_db()
        limit = request.args.get('limit', 50, type=int)
        unread_only = request.args.get('unread', 'false').lower() == 'true'
        notifications = db.get_notifications(limit=limit, unread_only=unread_only)
        unread_count = db.get_unread_count()
        return jsonify({'notifications': notifications, 'unread_count': unread_count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/notifications/read', methods=['POST'])
def api_orb_mark_read():
    """Mark notification(s) as read."""
    try:
        db = _get_orb_db()
        nid = request.json.get('id')
        if nid == 'all':
            db.mark_all_read()
        elif nid:
            db.mark_read(nid)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/book-pnl-series')
def api_orb_book_pnl_series():
    """Return the engine's rolling book-P&L samples for the in-app chart.

    The engine appends one {ts, pnl_inr, realized, unrealized} sample
    per monitor tick (~30s). Series resets at 09:14 initialize_day().

    Response:
      {
        "series": [{"ts": "...", "pnl_inr": ...}, ...],
        "threshold_soft_inr": -7500,
        "threshold_hard_inr": -15000,
        "current": <last pnl or null>,
      }
    """
    try:
        engine = _get_orb_engine()
        series = list(getattr(engine, '_book_pnl_history', []) or [])
        cfg = ORB_DEFAULTS
        return jsonify({
            'series': series,
            'threshold_soft_inr': int(cfg.get('book_drawdown_soft_inr', -7500)),
            'threshold_hard_inr': int(cfg.get('book_drawdown_hard_inr', -15000)),
            'current': series[-1]['pnl_inr'] if series else None,
        })
    except Exception as e:
        logger.error(f"[API] /api/orb/book-pnl-series error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/tv-pine')
def api_orb_tv_pine():
    """Generate a ready-to-paste Pine Script v5 indicator that plots the
    running unrealized P&L of the currently open ORB book.

    Output: JSON `{pine_script, positions, generated_at}`.

    Paste the `pine_script` string into TradingView -> Pine Editor ->
    Add to chart. Works best on a 1-min chart (NIFTY as the host
    symbol is fine). Horizontal lines mark the soft (-Rs 7.5K) and
    hard (-Rs 15K) drawdown-cut thresholds plus a +Rs 30K upside
    reference line."""
    try:
        db = _get_orb_db()
        open_positions = db.get_open_positions() or []
        if not open_positions:
            return jsonify({
                'pine_script': '// No open ORB positions right now — nothing to chart.',
                'positions': [],
                'generated_at': datetime.now().isoformat(),
            })

        # Config thresholds (fall back to defaults)
        cfg = ORB_DEFAULTS
        soft = int(cfg.get('book_drawdown_soft_inr', -7500))
        hard = int(cfg.get('book_drawdown_hard_inr', -15000))

        # Build variable blocks
        q_lines = []
        e_lines = []
        p_lines = []
        pnl_terms = []
        for i, pos in enumerate(open_positions, start=1):
            sym = pos['instrument']
            qty = int(pos['qty'])
            entry = float(pos['entry_price'])
            direction = pos['direction']
            # Signed quantity for LONG vs SHORT: pnl = signed_qty * (current - entry)
            # For SHORT we want pnl = qty * (entry - current), so the term is qty*(e-p).
            # For LONG we want pnl = qty * (current - entry), so the term is qty*(p-e).
            q_lines.append(f"// {sym} ({direction})\nq{i} = {qty}")
            e_lines.append(f"e{i} = {entry:.2f}")
            p_lines.append(
                f'p{i:<2} = request.security("NSE:{sym}", "1", close, '
                f'lookahead=barmerge.lookahead_off)'
            )
            if direction == 'SHORT':
                pnl_terms.append(f"q{i}*(e{i}-p{i})")
            else:
                pnl_terms.append(f"q{i}*(p{i}-e{i})")

        q_block = "\n".join(q_lines)
        e_block = "\n".join(e_lines)
        p_block = "\n".join(p_lines)
        pnl_expr = " + ".join(pnl_terms)

        script = f"""//@version=5
indicator("ORB Book P&L - Live", overlay=false, precision=0)

// =====================================================================
// ORB open book snapshot at {datetime.now().strftime('%Y-%m-%d %H:%M IST')}
// Auto-generated by /api/orb/tv-pine — regenerate when positions change.
// =====================================================================

{q_block}

{e_block}

// Live LTPs (1-min)
{p_block}

// Running book P&L
pnl = {pnl_expr}

// Plot
plot(pnl, title="Book P&L (Rs)", color=pnl >= 0 ? color.new(color.green, 0) : color.new(color.red, 0), linewidth=2)

// Reference lines
hline(0,      "Breakeven",       color=color.new(color.gray, 0),    linestyle=hline.style_dashed)
hline({soft}, "Soft cut",        color=color.new(color.orange, 0),  linestyle=hline.style_dashed)
hline({hard}, "Hard cut",        color=color.new(color.red, 0),     linestyle=hline.style_solid)
hline(15000,  "+Rs 15K",         color=color.new(color.green, 40),  linestyle=hline.style_dotted)
hline(30000,  "+Rs 30K",         color=color.new(color.green, 0),   linestyle=hline.style_dotted)

// Label last value on the right
var label lbl = na
label.delete(lbl)
lbl := label.new(bar_index, pnl, text="Rs " + str.tostring(pnl, "#,##0"), style=label.style_label_left, color=pnl >= 0 ? color.green : color.red, textcolor=color.white, size=size.normal)
"""

        return jsonify({
            'pine_script': script,
            'positions': [
                {
                    'instrument': p['instrument'],
                    'direction': p['direction'],
                    'qty': p['qty'],
                    'entry_price': p['entry_price'],
                } for p in open_positions
            ],
            'generated_at': datetime.now().isoformat(),
            'threshold_soft_inr': soft,
            'threshold_hard_inr': hard,
        })
    except Exception as e:
        logger.error(f"[API] /api/orb/tv-pine error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ORB Scheduled Jobs ----------------------------------------------------------


def _orb_initialize_day():
    """9:14 AM Mon-Fri: Compute CPR from Kite API, reset OR state for all 15 stocks.

    CRITICAL: MUST use the singleton via _get_orb_engine(), NOT a fresh
    ORBLiveEngine instance. update_or() reads from the singleton's
    _or_state dict — a throwaway instance would populate in-memory state
    that the scheduled update_or never sees, causing silent skip of all
    symbols (observed 2026-04-21 — 0 trades despite scheduler firing)."""
    if not ORB_DEFAULTS.get('enabled', True):
        return
    try:
        engine = _get_orb_engine()
        engine.initialize_day()
        logger.info("[ORB] Day initialization complete (via Kite API)")
    except Exception as e:
        logger.error(f"[ORB] Day init error: {e}")


def _get_orb_engine():
    """Get a shared ORBLiveEngine instance."""
    if not hasattr(_get_orb_engine, '_instance'):
        from services.orb_live_engine import ORBLiveEngine
        _get_orb_engine._instance = ORBLiveEngine(ORB_DEFAULTS)
    return _get_orb_engine._instance


def _orb_update_or():
    """Every 1 min from 9:15-9:30: Update OR high/low from Kite API."""
    if not ORB_DEFAULTS.get('enabled', True):
        return
    try:
        engine = _get_orb_engine()
        engine.update_or()
    except Exception as e:
        logger.error(f"[ORB] OR update error: {e}")


def _orb_evaluate_signals():
    """ORB signal evaluation. Cron-aligned to fire at second=5 of every clean
    5-min boundary, gated to the post-OR window. OR15 closes at 09:30;
    the cron tick at 09:30:05 is a silent no-op (latest closed 5-min candle
    is 09:25-09:30, still inside OR), and the first real evaluation happens
    at 09:35:05 on the 09:30-09:35 candle close — the earliest possible
    post-OR breakout candle. Upper bound is 15:15 (1 min before EOD squareoff).
    2026-04-28: gate moved from 09:45 to 09:30 — the prior 'wait for first
    15-min RSI bar to settle' rationale was unnecessary, since RSI(14) on
    resampled 15-min bars draws from weeks of historical data; partial
    same-day bars are not a problem (confirmed 2026-04-27 — yesterday's
    pre-09:45 signals had valid RSI readings)."""
    now_t = datetime.now().time()
    if now_t < dtime(9, 30) or now_t >= dtime(15, 16):
        return
    if not ORB_DEFAULTS.get('enabled', True):
        return
    try:
        engine = _get_orb_engine()
        engine.evaluate_signals()
    except Exception as e:
        logger.error(f"[ORB] Signal eval error: {e}")


def _orb_monitor_positions():
    """Every 30 sec: Check SL/target on open positions via Kite LTP."""
    if not ORB_DEFAULTS.get('enabled', True):
        return
    try:
        engine = _get_orb_engine()
        engine.monitor_positions()
    except Exception as e:
        logger.error(f"[ORB] Position monitor error: {e}")


def _orb_midmorning_status():
    """10:30 Mon-Fri: Send snapshot of open positions, day P&L, margin.
    Routes through NotificationService so both email and WhatsApp fire."""
    if not ORB_DEFAULTS.get('enabled', True):
        return
    try:
        engine = _get_orb_engine()
        engine.send_midmorning_status()
    except Exception as e:
        logger.error(f"[ORB] midmorning status error: {e}")


def _orb_activate_trail():
    """14:30: V9t_lock50 — lock 50% of profit on each open position
    (move SL to entry + 0.5*gain). Positions ride until SL hit or 15:18 hard EOD.
    Backtest winner: Calmar 676 vs force-close baseline 281."""
    if not ORB_DEFAULTS.get('enabled', True):
        return
    try:
        engine = _get_orb_engine()
        engine.activate_trail_lock50()
    except Exception as e:
        logger.error(f"[ORB] 14:30 trail activation error: {e}")


def _orb_eod_squareoff():
    """15:16 sharp: Hard close any remaining open ORB positions via our
    own LIMIT orders. Runs ~4 min before Zerodha MIS auto-squareoff
    (15:20-15:25) to avoid the Rs 59/trade auto-squareoff charge.
    With 11 positions that fee is Rs 649/day — over a year on 60-70
    triggered days it adds up to ~Rs 45K, meaningful versus day P&L."""
    if not ORB_DEFAULTS.get('enabled', True):
        return
    try:
        engine = _get_orb_engine()
        engine.eod_squareoff()
    except Exception as e:
        logger.error(f"[ORB] EOD squareoff error: {e}")


def _orb_eod_report():
    """15:25 PM Mon-Fri: Send daily EOD summary report."""
    if not ORB_DEFAULTS.get('enabled', True) or not ORB_DEFAULTS.get('notify_eod_report', True):
        return
    try:
        from services.orb_live_engine import ORBLiveEngine
        engine = ORBLiveEngine(ORB_DEFAULTS)
        engine.generate_eod_report()
        logger.info("[ORB] EOD report sent")
    except Exception as e:
        logger.error(f"[ORB] EOD report error: {e}")


def _orb_daily_backtest():
    """15:45 PM Mon-Fri: Re-simulate today's ORB signals from raw Kite data
    and persist to orb_backtest.db for the Performance page."""
    if not ORB_DEFAULTS.get('enabled', True):
        return
    try:
        from services.orb_daily_backtest import run_backtest
        out = run_backtest()
        logger.info(
            f"[ORB-BT] {out['run_date']} stored: taken={out['trades_taken']} "
            f"blocked={out['signals_blocked']} net=Rs{out['net_pnl_inr']:+.0f}"
        )
    except Exception as e:
        logger.error(f"[ORB-BT] daily backtest error: {e}", exc_info=True)


# Register ORB scheduled jobs
try:
    scheduler.add_job(
        _orb_initialize_day,
        'cron', day_of_week='mon-fri', hour=9, minute=14,
        id='orb_init_day', replace_existing=True,
    )
    scheduler.add_job(
        _orb_update_or,
        'cron', day_of_week='mon-fri', hour=9, minute='15-32',
        id='orb_update_or', replace_existing=True,
    )
    scheduler.add_job(
        _orb_evaluate_signals,
        # Cron-aligned to clean 5-min boundaries with second=5 — fires
        # ~5s after each 5-min candle close. Small buffer ensures Kite's
        # historical_data API has reliably published the just-closed bar
        # (typical publish latency 1-3s). 25s tighter than the original
        # :30s grace; first eval per day moves from 09:45:30 to 09:45:05.
        # Restricted to Mon-Fri market hours (9-15) at the cron level;
        # the function itself gates the precise 09:45-15:15 window.
        'cron', day_of_week='mon-fri', hour='9-15', minute='*/5', second=5,
        id='orb_eval_signals', replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        _orb_monitor_positions,
        'interval', seconds=30,
        id='orb_monitor_pos', replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        _orb_midmorning_status,
        'cron', day_of_week='mon-fri', hour=10, minute=30,
        id='orb_midmorning_status', replace_existing=True,
    )
    scheduler.add_job(
        _orb_activate_trail,
        'cron', day_of_week='mon-fri', hour=14, minute=30,
        id='orb_activate_trail', replace_existing=True,
    )
    scheduler.add_job(
        _orb_eod_squareoff,
        'cron', day_of_week='mon-fri', hour=15, minute=16,
        id='orb_eod_squareoff', replace_existing=True,
    )
    scheduler.add_job(
        _orb_eod_report,
        'cron', day_of_week='mon-fri', hour=15, minute=25,
        id='orb_eod_report', replace_existing=True,
    )
    scheduler.add_job(
        _orb_daily_backtest,
        'cron', day_of_week='mon-fri', hour=15, minute=45,
        id='orb_daily_backtest', replace_existing=True,
    )
    logger.info(
        "ORB scheduled jobs registered: "
        "init(9:14), OR update(9:15-9:29), signal eval(5min), "
        "position monitor(30s), midmorning status(10:30), "
        "V9t_lock50 trail(14:30), hard EOD squareoff(15:16), "
        "EOD report(15:25), daily backtest(15:45)"
    )
except Exception as e:
    logger.warning(f"Could not register ORB scheduled jobs: {e}")


# --- NAS Performance Report ---------------------------------------------------

@app.route('/nas/report')
def nas_report():
    """NAS Performance Report — all 4 systems side by side."""
    return render_template(
        'nas_report.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/api/nas/report-data')
def api_nas_report_data():
    """Aggregated report data across all 4 NAS trading systems."""
    import sqlite3 as _sqlite3

    systems = {
        'OTM':  'backtest_data/nas_trading.db',
        'ATM':  'backtest_data/nas_atm_trading.db',
        'ATM2': 'backtest_data/nas_atm2_trading.db',
        'ATM4': 'backtest_data/nas_atm4_trading.db',
        '916-OTM':  'backtest_data/nas_916_otm_trading.db',
        '916-ATM':  'backtest_data/nas_916_atm_trading.db',
        '916-ATM2': 'backtest_data/nas_916_atm2_trading.db',
        '916-ATM4': 'backtest_data/nas_916_atm4_trading.db',
    }

    result = {'systems': {}, 'daily_snapshots': {}}

    for sys_name, db_path in systems.items():
        try:
            conn = _sqlite3.connect(db_path)
            conn.row_factory = _sqlite3.Row

            # Table prefix
            # OTM variants use nas_* tables; ATM variants use nas_atm_* tables.
            prefix = 'nas_' if sys_name in ('OTM', '916-OTM') else 'nas_atm_'

            # Get all positions
            positions = conn.execute(
                f"SELECT * FROM {prefix}positions ORDER BY id"
            ).fetchall()
            positions = [dict(r) for r in positions]

            # Get all trades
            trades = conn.execute(
                f"SELECT * FROM {prefix}trades ORDER BY id"
            ).fetchall()
            trades = [dict(r) for r in trades]

            # System summary
            # Compute P&L from positions (trades table may have stale data)
            closed_positions = [p for p in positions if p.get('status') == 'CLOSED' and p.get('exit_price')]
            pos_pnls = []
            for p in closed_positions:
                pnl = (p.get('entry_price', 0) - (p.get('exit_price', 0) or 0)) * (p.get('qty', 0) or 75)
                pos_pnls.append(pnl)

            # Group by strangle for trade-level stats
            strangle_pnls = {}
            for p in closed_positions:
                sid = p.get('strangle_id', 0)
                pnl = (p.get('entry_price', 0) - (p.get('exit_price', 0) or 0)) * (p.get('qty', 0) or 75)
                strangle_pnls[sid] = strangle_pnls.get(sid, 0) + pnl

            trade_pnls = list(strangle_pnls.values())
            total_trades = len(trade_pnls)
            wins = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p <= 0]

            result['systems'][sys_name] = {
                'total_trades': total_trades,
                'total_pnl': round(sum(pos_pnls), 2),
                'avg_pnl': round(sum(trade_pnls) / total_trades, 2) if total_trades else 0,
                'win_rate': round(len(wins) / total_trades * 100, 1) if total_trades else 0,
                'winners': len(wins),
                'losers': len(losses),
                'max_win': round(max(trade_pnls), 2) if trade_pnls else 0,
                'max_loss': round(min(trade_pnls), 2) if trade_pnls else 0,
                'profit_factor': round(
                    sum(wins) / abs(sum(losses)), 2
                ) if losses and sum(losses) != 0 else 0,
                'trades': trades,
                'positions': positions,
            }
            conn.close()
        except Exception as e:
            result['systems'][sys_name] = {
                'error': str(e),
                'total_trades': 0,
                'positions': [],
                'trades': [],
            }

    # Build daily snapshots (all systems merged by date)
    all_dates = set()
    for sys_name, data in result['systems'].items():
        for pos in data.get('positions', []):
            entry_date = (pos.get('entry_time') or '')[:10]
            if entry_date:
                all_dates.add(entry_date)

    daily = {}
    for d in sorted(all_dates):
        daily[d] = {}
        for sys_name, data in result['systems'].items():
            day_positions = [
                p for p in data.get('positions', [])
                if (p.get('entry_time') or '')[:10] == d
            ]
            day_trades = [
                t for t in data.get('trades', [])
                if (t.get('trade_date') or '') == d
            ]
            # Compute day P&L from positions (not trades table)
            day_pnl = 0
            for p in day_positions:
                if p.get('exit_price'):
                    day_pnl += (p.get('entry_price', 0) - (p.get('exit_price', 0) or 0)) * (p.get('qty', 0) or 75)
            daily[d][sys_name] = {
                'positions': day_positions,
                'trades': day_trades,
                'day_pnl': round(day_pnl, 2),
                'trade_count': len(day_positions),
            }

    result['daily_snapshots'] = daily

    # Remove raw positions/trades from system summary (too large for JSON)
    for sys_name in result['systems']:
        result['systems'][sys_name].pop('positions', None)
        result['systems'][sys_name].pop('trades', None)

    return jsonify(result)


@app.route('/api/orb/backtest')
def api_orb_backtest():
    """Return stored ORB daily backtest results.

    Query params:
      - date=YYYY-MM-DD    specific run, latest if omitted
      - list=1             return [summary, summary, ...] for recent runs instead
    """
    try:
        from services.orb_daily_backtest import get_backtest_run, list_backtest_runs
        if request.args.get('list'):
            return jsonify(list_backtest_runs(limit=60))
        d = request.args.get('date') or None
        run = get_backtest_run(run_date=d)
        if not run:
            return jsonify({'error': 'no backtest runs yet'}), 404
        return jsonify(run)
    except Exception as e:
        logger.error(f"[API] /api/orb/backtest error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/live-daily')
def api_orb_live_daily():
    """Return all closed ORB cash live positions, grouped by trade_date.

    Each entry: { trade_date, trades_count, winners, losers, daily_pnl_inr,
    trades: [position_dict, ...] }, ordered newest first.
    """
    try:
        from services.orb_db import get_orb_db
        db = get_orb_db()
        with db.db_lock:
            conn = db._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM orb_positions "
                    "WHERE status='CLOSED' "
                    "ORDER BY trade_date DESC, exit_time ASC"
                ).fetchall()
                rows = [dict(r) for r in rows]
            finally:
                conn.close()

        days = {}
        for r in rows:
            d = r.get('trade_date') or ''
            if not d:
                continue
            bucket = days.setdefault(d, {
                'trade_date': d,
                'trades_count': 0,
                'winners': 0,
                'losers': 0,
                'daily_pnl_inr': 0.0,
                'trades': [],
            })
            pnl = r.get('pnl_inr') or 0
            bucket['trades_count'] += 1
            bucket['daily_pnl_inr'] += pnl
            if pnl > 0:
                bucket['winners'] += 1
            else:
                bucket['losers'] += 1
            bucket['trades'].append(r)

        out = sorted(days.values(), key=lambda b: b['trade_date'], reverse=True)
        for b in out:
            b['daily_pnl_inr'] = round(b['daily_pnl_inr'], 2)

        # Top-level summary across all live closed trades
        total_pnl = round(sum(b['daily_pnl_inr'] for b in out), 2)
        total_trades = sum(b['trades_count'] for b in out)
        total_wins = sum(b['winners'] for b in out)

        return jsonify({
            'days': out,
            'summary': {
                'total_trades': total_trades,
                'winners': total_wins,
                'losers': total_trades - total_wins,
                'win_rate': round(total_wins / total_trades * 100, 1) if total_trades else 0,
                'total_pnl_inr': total_pnl,
                'active_days': len(out),
            },
        })
    except Exception as e:
        logger.error(f"[API] /api/orb/live-daily error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/nas/debug-counts')
def api_nas_debug_counts():
    """Quick row-count check across all 8 NAS DBs — for diagnosing empty reports."""
    import sqlite3 as _sqlite3
    pairs = [
        ('OTM', 'backtest_data/nas_trading.db', 'nas_'),
        ('ATM', 'backtest_data/nas_atm_trading.db', 'nas_atm_'),
        ('ATM2', 'backtest_data/nas_atm2_trading.db', 'nas_atm_'),
        ('ATM4', 'backtest_data/nas_atm4_trading.db', 'nas_atm_'),
        ('916-OTM', 'backtest_data/nas_916_otm_trading.db', 'nas_'),
        ('916-ATM', 'backtest_data/nas_916_atm_trading.db', 'nas_atm_'),
        ('916-ATM2', 'backtest_data/nas_916_atm2_trading.db', 'nas_atm_'),
        ('916-ATM4', 'backtest_data/nas_916_atm4_trading.db', 'nas_atm_'),
    ]
    out = {}
    for name, db, pfx in pairs:
        try:
            if not os.path.exists(db):
                out[name] = {'error': 'MISSING', 'path': db}; continue
            c = _sqlite3.connect(db)
            tabs = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            tbl = pfx + 'positions'
            if tbl not in tabs:
                out[name] = {'error': 'missing-table', 'expected': tbl, 'tables': tabs}
                c.close(); continue
            tot = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            cl = c.execute(
                f"SELECT COUNT(*) FROM {tbl} WHERE status='CLOSED' "
                f"AND exit_price IS NOT NULL"
            ).fetchone()[0]
            statuses = dict(c.execute(
                f"SELECT status, COUNT(*) FROM {tbl} GROUP BY status"
            ).fetchall())
            sample = c.execute(
                f"SELECT id, status, entry_price, exit_price, qty, entry_time "
                f"FROM {tbl} ORDER BY id DESC LIMIT 2"
            ).fetchall()
            out[name] = {
                'positions_total': tot,
                'positions_closed_with_exit': cl,
                'status_breakdown': statuses,
                'sample_recent': [
                    {'id': r[0], 'status': r[1], 'entry_price': r[2],
                     'exit_price': r[3], 'qty': r[4], 'entry_time': r[5]}
                    for r in sample
                ],
                'db_path': db,
                'prefix_used': pfx,
            }
            c.close()
        except Exception as e:
            out[name] = {'error': str(e), 'path': db, 'prefix_used': pfx}
    return jsonify(out)


@app.route('/api/validation/<report_type>/latest')
def api_validation_latest(report_type):
    """Return the most recent validator report (premarket | eod) as JSON."""
    if report_type not in ('premarket', 'eod'):
        return jsonify({'error': 'report_type must be premarket or eod'}), 400
    try:
        from services.system_validator import get_latest
        rep = get_latest(report_type)
        if rep is None:
            return jsonify({'error': f'no {report_type} report yet'}), 404
        return jsonify(rep)
    except Exception as e:
        logger.error(f"[API] /api/validation/{report_type}/latest error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/premarket/brief/latest')
def api_premarket_brief_latest():
    """Return the most recent pre-market brief JSON."""
    try:
        from services.premarket_brief import get_latest_brief
        b = get_latest_brief()
        if b is None:
            return jsonify({'error': 'no brief generated yet'}), 404
        return jsonify(b)
    except Exception as e:
        logger.error(f"[API] /api/premarket/brief/latest error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/premarket/brief/raw')
def api_premarket_brief_raw():
    """Build a fresh brief without sending email — for cloud routine to fetch."""
    try:
        from services.premarket_brief import build_brief
        return jsonify(build_brief())
    except Exception as e:
        logger.error(f"[API] /api/premarket/brief/raw error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/premarket/brief/run', methods=['POST'])
def api_premarket_brief_run():
    """Trigger a full brief run (build + persist + email) on demand."""
    try:
        from services.premarket_brief import run_premarket_brief
        return jsonify(run_premarket_brief())
    except Exception as e:
        logger.error(f"[API] /api/premarket/brief/run error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/premarket/brief/synthesized', methods=['POST'])
def api_premarket_brief_synthesized():
    """Receive synthesized headlines + narrative from the cloud Claude routine
    and dispatch the email. Body must be JSON with shape:
        { "headlines_synthesized": [{"tag": "POS", "text": "...", "source": "..."}, ...],
          "narrative_summary": "optional 1-2 sentence one-liner override" }
    """
    try:
        from services.premarket_brief import receive_synthesis_and_send
        body = request.get_json(silent=True) or {}
        result = receive_synthesis_and_send(body)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[API] /api/premarket/brief/synthesized error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/validation/<report_type>/run', methods=['POST'])
def api_validation_run(report_type):
    """Trigger a validator run on demand. Returns the freshly built report."""
    if report_type not in ('premarket', 'eod'):
        return jsonify({'error': 'report_type must be premarket or eod'}), 400
    try:
        from services.system_validator import run_premarket_check, run_eod_check
        fn = run_premarket_check if report_type == 'premarket' else run_eod_check
        rep = fn()
        return jsonify(rep)
    except Exception as e:
        logger.error(f"[API] /api/validation/{report_type}/run error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/orb/backtest/run', methods=['POST'])
def api_orb_backtest_run():
    """Trigger an ORB backtest for today (or provided date) and store results."""
    try:
        from services.orb_daily_backtest import run_backtest
        from datetime import date as _date
        body = request.get_json(silent=True) or {}
        date_arg = request.args.get('date') or body.get('date')
        run_date = _date.fromisoformat(date_arg) if date_arg else None
        out = run_backtest(run_date=run_date)
        return jsonify({
            'status': 'ok',
            'run_date': out['run_date'],
            'trades_taken': out['trades_taken'],
            'signals_blocked': out['signals_blocked'],
            'net_pnl_inr': out['net_pnl_inr'],
        })
    except Exception as e:
        logger.error(f"[API] /api/orb/backtest/run error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =============================================================================
# Holdings dashboard — live digest, daily snapshots, nightly meta / events crons
# =============================================================================

@app.route('/api/holdings/digest')
def api_holdings_digest():
    """Live digest — summary + movers + extremes + weekly + events + next event."""
    try:
        from services.holdings_dashboard import get_digest
        return jsonify(get_digest())
    except Exception as e:
        logger.error(f"[API] /api/holdings/digest error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/holdings/snapshot')
def api_holdings_snapshot():
    """Return a stored daily snapshot. ?date=YYYY-MM-DD or latest if omitted."""
    try:
        from services.holdings_dashboard import get_snapshot
        d = request.args.get('date') or None
        snap = get_snapshot(snap_date=d)
        if not snap:
            return jsonify({'error': 'no snapshot'}), 404
        return jsonify(snap)
    except Exception as e:
        logger.error(f"[API] /api/holdings/snapshot error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/holdings/snapshots')
def api_holdings_snapshots():
    """List recent snapshot summaries, newest first."""
    try:
        from services.holdings_dashboard import list_snapshots
        limit = int(request.args.get('limit', 120))
        return jsonify(list_snapshots(limit=limit))
    except Exception as e:
        logger.error(f"[API] /api/holdings/snapshots error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/holdings/refresh', methods=['POST'])
def api_holdings_refresh():
    """Trigger meta + events + snapshot refresh on demand (admin)."""
    try:
        from services.holdings_dashboard import (
            refresh_holdings_meta, refresh_corporate_actions,
            capture_daily_snapshot,
        )
        body = request.get_json(silent=True) or {}
        parts = body.get('parts') or ['meta', 'events', 'snapshot']
        result = {}
        if 'meta' in parts:
            result['meta'] = refresh_holdings_meta()
        if 'events' in parts:
            result['events'] = refresh_corporate_actions()
        if 'snapshot' in parts:
            result['snapshot'] = capture_daily_snapshot()
        return jsonify({'status': 'ok', **result})
    except Exception as e:
        logger.error(f"[API] /api/holdings/refresh error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _holdings_refresh_meta_job():
    """06:30 Mon-Fri — Refresh per-symbol 52w hi/lo + 5d/20d moves."""
    try:
        from services.holdings_dashboard import refresh_holdings_meta
        refresh_holdings_meta()
    except Exception as e:
        logger.error(f"[Holdings-cron] meta error: {e}", exc_info=True)


def _holdings_refresh_events_job():
    """07:00 Mon-Fri — Fetch upcoming corporate actions from NSE."""
    try:
        from services.holdings_dashboard import refresh_corporate_actions
        refresh_corporate_actions()
    except Exception as e:
        logger.error(f"[Holdings-cron] events error: {e}", exc_info=True)


def _holdings_capture_snapshot_job():
    """16:00 Mon-Fri — Capture post-close daily snapshot for history view."""
    try:
        from services.holdings_dashboard import capture_daily_snapshot
        capture_daily_snapshot()
    except Exception as e:
        logger.error(f"[Holdings-cron] snapshot error: {e}", exc_info=True)


try:
    scheduler.add_job(_holdings_refresh_meta_job,
                      'cron', day_of_week='mon-fri', hour=6, minute=30,
                      id='holdings_meta', replace_existing=True)
    scheduler.add_job(_holdings_refresh_events_job,
                      'cron', day_of_week='mon-fri', hour=7, minute=0,
                      id='holdings_events', replace_existing=True)
    scheduler.add_job(_holdings_capture_snapshot_job,
                      'cron', day_of_week='mon-fri', hour=16, minute=0,
                      id='holdings_snapshot', replace_existing=True)
    logger.info("Holdings cron jobs registered: meta(06:30), events(07:00), snapshot(16:00)")
except Exception as e:
    logger.warning(f"Could not register Holdings cron jobs: {e}")


# =============================================================================
# Options Data Manager — Index Option Chain Downloader
# =============================================================================

@app.route('/api/options/capture', methods=['POST'])
def api_options_capture():
    """Manually trigger option chain capture for all indices."""
    task_id = f"opts_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _nas_tasks[task_id] = {'status': 'running'}

    def _run_capture(tid):
        try:
            from services.options_data_manager import OptionsDataManager
            manager = OptionsDataManager()
            results = manager.capture_all()
            _nas_tasks[tid] = {'status': 'completed', 'results': results}
        except Exception as e:
            logger.error(f"Options capture error: {e}")
            _nas_tasks[tid] = {'status': 'error', 'error': str(e)}

    scheduler.add_job(_run_capture, args=[task_id], id=f'opts_{task_id}',
                      replace_existing=True)
    return jsonify({'task_id': task_id, 'status': 'queued'})


@app.route('/api/options/capture/status/<task_id>')
def api_options_capture_status(task_id):
    """Poll option chain capture status."""
    task = _nas_tasks.get(task_id, {'status': 'unknown'})
    return jsonify(task)


@app.route('/api/options/stats')
def api_options_stats():
    """Get options database statistics."""
    try:
        from services.options_data_manager import get_options_db
        db = get_options_db()
        return jsonify(db.get_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/plans/future')
def api_future_plans():
    """Serve the structured future-plans log from data/future_plans.json.
    File is editable without code changes — plan authors just append to
    the plans[] array."""
    import json as _json_fp
    from pathlib import Path
    try:
        path = Path(__file__).resolve().parent / 'data' / 'future_plans.json'
        if not path.exists():
            return jsonify({'plans': []})
        with open(path, 'r', encoding='utf-8') as f:
            return jsonify(_json_fp.load(f))
    except Exception as e:
        logger.error(f'[PLANS] load error: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/options/coverage')
def api_options_coverage():
    """Per-session summary of captured options data. Returns:
    - cumulative stats (size, rows, date range, sessions)
    - per-day breakdown: date, rows, symbols, first/last snapshot, per-index"""
    try:
        import os
        from services.options_data_manager import get_options_db, OPTIONS_DB_PATH
        db = get_options_db()
        with db.db_lock:
            conn = db._get_conn()
            try:
                daily_rows = conn.execute("""
                    SELECT DATE(snapshot_time) d,
                           COUNT(*) n,
                           COUNT(DISTINCT tradingsymbol) syms,
                           MIN(snapshot_time) first_ts,
                           MAX(snapshot_time) last_ts
                    FROM option_chain
                    GROUP BY d
                    ORDER BY d DESC
                """).fetchall()
                per_idx_by_day = conn.execute("""
                    SELECT DATE(snapshot_time) d, symbol, COUNT(*) n
                    FROM option_chain
                    GROUP BY d, symbol
                """).fetchall()
                cum = conn.execute("""
                    SELECT COUNT(*), COUNT(DISTINCT tradingsymbol),
                           COUNT(DISTINCT DATE(snapshot_time)),
                           MIN(DATE(snapshot_time)), MAX(DATE(snapshot_time))
                    FROM option_chain
                """).fetchone()
                spot_rows = conn.execute("SELECT COUNT(*) FROM underlying_spot").fetchone()[0]
                try:
                    ohlc_rows = conn.execute("SELECT COUNT(*) FROM option_ohlc").fetchone()[0]
                except Exception:
                    ohlc_rows = 0
            finally:
                conn.close()

        # Build per-index map by day
        by_day_idx: dict = {}
        for r in per_idx_by_day:
            d = dict(r)
            by_day_idx.setdefault(d['d'], {})[d['symbol']] = d['n']

        size_mb = os.path.getsize(OPTIONS_DB_PATH) / (1024 * 1024) if os.path.exists(OPTIONS_DB_PATH) else 0

        sessions = []
        for r in daily_rows:
            d = dict(r)
            first_t = (d['first_ts'] or '')[11:19]
            last_t = (d['last_ts'] or '')[11:19]
            sessions.append({
                'date': d['d'],
                'rows': d['n'],
                'symbols': d['syms'],
                'first_snapshot': first_t,
                'last_snapshot': last_t,
                'per_index': by_day_idx.get(d['d'], {}),
                'status': 'ok' if d['n'] > 0 else 'failed',
            })

        return jsonify({
            'cumulative': {
                'rows': cum[0],
                'symbols': cum[1],
                'sessions': cum[2],
                'date_min': cum[3],
                'date_max': cum[4],
                'spot_rows': spot_rows,
                'ohlc_rows': ohlc_rows,
                'size_mb': round(size_mb, 2),
            },
            'sessions': sessions,
            'granularity': '1-min snapshots (option_chain + underlying_spot)',
            'capture_window': '09:20 - 15:30 IST Mon-Fri',
        })
    except Exception as e:
        logger.error(f"[OPTIONS] coverage error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/options/chain/<symbol>')
def api_options_chain(symbol):
    """Get latest option chain for a symbol (NIFTY/BANKNIFTY/SENSEX)."""
    try:
        from services.options_data_manager import get_options_db
        db = get_options_db()
        expiry = request.args.get('expiry')
        snap_time = request.args.get('time')
        chain = db.get_option_chain(symbol.upper(), snap_time, expiry)
        return jsonify({'symbol': symbol.upper(), 'count': len(chain), 'data': chain})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/options/snapshots/<symbol>')
def api_options_snapshots(symbol):
    """Get available snapshot times for a symbol."""
    try:
        from services.options_data_manager import get_options_db
        db = get_options_db()
        trade_date = request.args.get('date')
        snapshots = db.get_available_snapshots(symbol.upper(), trade_date)
        return jsonify(snapshots)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/options-data/summary')
def api_options_data_summary():
    """Summary of captured options data: snapshots, dates, symbols, IV coverage."""
    try:
        from services.options_data_manager import get_options_db
        db = get_options_db()
        return jsonify(db.get_summary())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/options/download-log')
def api_options_download_log():
    """Get recent download log entries."""
    try:
        from services.options_data_manager import get_options_db
        db = get_options_db()
        with db.db_lock:
            conn = db._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM download_log ORDER BY created_at DESC LIMIT 50"
                ).fetchall()
                return jsonify([dict(r) for r in rows])
            finally:
                conn.close()
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- Options Data Scheduled Jobs ----

def _options_capture_job():
    """Every 1 min Mon-Fri 9:20-15:20 — Capture index option chains."""
    now = datetime.now()
    market_start = now.replace(hour=9, minute=20, second=0)
    market_end = now.replace(hour=15, minute=30, second=0)
    if now < market_start or now > market_end:
        return  # Outside market hours
    try:
        from services.options_data_manager import OptionsDataManager
        manager = OptionsDataManager()
        results = manager.capture_all()
        total = sum(r.get('instruments_captured', 0) for r in results)
        errors = [r for r in results if r.get('error')]
        if errors:
            logger.warning(f"[OPTIONS] Capture partial: {total} instruments, "
                           f"{len(errors)} errors: {[e['error'] for e in errors]}")
    except Exception as e:
        logger.error(f"[OPTIONS] Capture job error: {e}")


# Capture option chain every 1 minute during market hours (9:20-15:20)
try:
    scheduler.add_job(
        _options_capture_job,
        'cron', day_of_week='mon-fri',
        hour='9-15', minute='*',
        id='options_capture_1min', replace_existing=True,
    )
    logger.info(
        "Options data capture scheduled every 1 min Mon-Fri 9:00-15:59 "
        "(NIFTY + BANKNIFTY + SENSEX, with IV computation)"
    )
except Exception as e:
    logger.warning(f"Could not register options capture job: {e}")


# ---- Daily Options Capture EOD Summary ----

def _options_eod_summary():
    """15:35 Mon-Fri: Acknowledge today's capture + cumulative DB stats.
    Sends email + WhatsApp via NotificationService. Priority 'critical' if 0 rows today."""
    now = datetime.now()
    if now.weekday() >= 5:
        return
    try:
        import os
        from services.options_data_manager import get_options_db, OPTIONS_DB_PATH
        from services.notifications import get_notification_service

        db = get_options_db()
        today_str = now.strftime('%Y-%m-%d')

        with db.db_lock:
            conn = db._get_conn()
            try:
                today_rows = conn.execute(
                    "SELECT COUNT(*) FROM option_chain WHERE DATE(snapshot_time)=?",
                    (today_str,)
                ).fetchone()[0]
                today_syms = conn.execute(
                    "SELECT COUNT(DISTINCT tradingsymbol) FROM option_chain WHERE DATE(snapshot_time)=?",
                    (today_str,)
                ).fetchone()[0]
                today_range = conn.execute(
                    "SELECT MIN(snapshot_time), MAX(snapshot_time) FROM option_chain WHERE DATE(snapshot_time)=?",
                    (today_str,)
                ).fetchone()
                total_rows = conn.execute("SELECT COUNT(*) FROM option_chain").fetchone()[0]
                total_syms = conn.execute("SELECT COUNT(DISTINCT tradingsymbol) FROM option_chain").fetchone()[0]
                date_range = conn.execute(
                    "SELECT MIN(DATE(snapshot_time)), MAX(DATE(snapshot_time)), "
                    "COUNT(DISTINCT DATE(snapshot_time)) FROM option_chain"
                ).fetchone()
                spot_rows = conn.execute("SELECT COUNT(*) FROM underlying_spot").fetchone()[0]
                try:
                    ohlc_rows = conn.execute("SELECT COUNT(*) FROM option_ohlc").fetchone()[0]
                except Exception:
                    ohlc_rows = 0
                per_index = conn.execute(
                    "SELECT symbol, COUNT(*) n FROM option_chain "
                    "WHERE DATE(snapshot_time)=? GROUP BY symbol ORDER BY n DESC",
                    (today_str,)
                ).fetchall()
            finally:
                conn.close()

        size_mb = os.path.getsize(OPTIONS_DB_PATH) / (1024 * 1024) if os.path.exists(OPTIONS_DB_PATH) else 0

        if today_rows == 0:
            title = f'Options Capture FAILED -- {today_str}'
            priority = 'critical'
        else:
            title = f'Options Capture OK {today_str} -- {today_rows:,} rows / {today_syms} symbols'
            priority = 'normal'

        per_idx_str = ', '.join(f'{dict(r)["symbol"]}={dict(r)["n"]:,}' for r in per_index) or '-'
        lines = [
            f'Today ({today_str}):',
            f'  rows        : {today_rows:,}',
            f'  symbols     : {today_syms:,}',
            f'  time range  : {today_range[0] or "-"}  to  {today_range[1] or "-"}',
            f'  per-index   : {per_idx_str}',
            '',
            'Cumulative DB:',
            f'  option_chain    : {total_rows:,} rows, {total_syms:,} distinct symbols',
            f'  underlying_spot : {spot_rows:,} rows',
            f'  option_ohlc     : {ohlc_rows:,} rows',
            f'  date coverage   : {date_range[0]} to {date_range[1]}  ({date_range[2]} sessions)',
            f'  granularity     : 1-min snapshots (option_chain + underlying_spot)',
            f'  file size       : {size_mb:.1f} MB',
        ]
        message = '\n'.join(lines)

        ns = get_notification_service()
        ns.send_alert('options_capture_summary', title, message, priority=priority)
        logger.info(
            f'[OPTIONS] EOD summary sent: today={today_rows:,} rows, '
            f'total={total_rows:,} rows, {size_mb:.1f}MB, {date_range[2]} sessions'
        )
    except Exception as e:
        logger.error(f'[OPTIONS] EOD summary error: {e}', exc_info=True)

try:
    scheduler.add_job(
        _options_eod_summary,
        'cron', day_of_week='mon-fri', hour=15, minute=35,
        id='options_eod_summary', replace_existing=True,
    )
    logger.info('Options capture EOD summary scheduled: 15:35 Mon-Fri')
except Exception as e:
    logger.warning(f'Could not register options EOD summary job: {e}')


# ---- Daily Instrument Dump (for historical backfill) ----

def _instruments_dump_job():
    """9:20 AM Mon-Fri — Dump current NFO/BFO instruments for historical backfill."""
    try:
        from backfill_options_data import init_backfill_tables, dump_instruments
        from services.kite_service import get_kite
        init_backfill_tables()
        kite = get_kite()
        count = dump_instruments(kite)
        logger.info(f"[OPTIONS] Dumped {count} instruments to archive")
    except Exception as e:
        logger.error(f"[OPTIONS] Instrument dump error: {e}")

try:
    scheduler.add_job(
        _instruments_dump_job,
        'cron', day_of_week='mon-fri', hour=9, minute=20,
        id='instruments_dump', replace_existing=True,
    )
    logger.info("Instruments dump scheduled: 9:20 AM Mon-Fri (NFO + BFO)")
except Exception as e:
    logger.warning(f"Could not register instruments dump job: {e}")


# =============================================================================
# Tactical Capital Pool Dashboard
# =============================================================================

@app.route('/tactical')
def tactical_dashboard():
    """Tactical Capital Pool Dashboard - no login required (read-only)."""
    return render_template(
        'tactical_dashboard.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/crash-recovery')
def crash_recovery():
    """Crash & Recovery showcase - how MQ handles wealth during market crashes."""
    return render_template(
        'crash_recovery.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/api/tactical/state')
def tactical_state():
    """Get full tactical pool state for dashboard."""
    from services.tactical_pool import TacticalPoolDB
    db = TacticalPoolDB()
    return jsonify(db.get_dashboard_state())


# =============================================================================
# IPO Research Report
# =============================================================================

@app.route('/ipo-research')
def ipo_research_report():
    """IPO Launch Strategy Research Report - no login required (read-only)."""
    return render_template(
        'ipo_strategy_report.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/breakout-v3')
def breakout_v3_dashboard():
    """Breakout V3 Multi-Strategy Dashboard - no login required (read-only)."""
    return render_template(
        'breakout_v3_dashboard.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/api/breakout-v3/results')
def api_breakout_v3_results():
    """Return cached V3 backtest results."""
    try:
        from services.breakout_v3_backtest import get_cached_results
        results = get_cached_results()
        if results:
            return jsonify(results)
        return jsonify({'error': 'No results cached. Click Run Backtest.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/breakout-v3/run', methods=['POST'])
def api_breakout_v3_run():
    """Start V3 backtest in background."""
    import threading
    task_id = f'bov3_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    task_status[task_id] = {
        'status': 'running', 'progress': 0, 'message': 'Starting...',
    }

    def _run(tid):
        try:
            from services.breakout_v3_backtest import run_v3_backtest

            def cb(pct, msg):
                task_status[tid].update(progress=pct, message=msg)

            results = run_v3_backtest(progress_callback=cb)
            task_status[tid] = {
                'status': 'complete', 'progress': 100,
                'message': 'Done', 'result': 'saved',
            }
        except Exception as e:
            logger.error(f"V3 backtest failed: {e}")
            task_status[tid] = {
                'status': 'error', 'progress': 0,
                'message': str(e), 'error': str(e),
            }

    threading.Thread(target=_run, args=(task_id,), daemon=True).start()
    return jsonify({'task_id': task_id})


@app.route('/api/breakout-v3/status/<task_id>')
def api_breakout_v3_status(task_id):
    """Check V3 backtest task progress."""
    status = task_status.get(task_id)
    if status:
        return jsonify(status)
    return jsonify({'error': 'Task not found'}), 404


# =============================================================================
# Combined MQ + V3 Routes
# =============================================================================

@app.route('/combined')
def combined_dashboard():
    """Combined MQ + V3 Dashboard - no login required (read-only)."""
    return render_template(
        'combined_mq_v3_dashboard.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/api/combined/results')
def api_combined_results():
    """Return cached combined backtest results."""
    try:
        results_path = Path('backtest_data/combined_mq_v3_results.json')
        if results_path.exists():
            import json
            with open(results_path) as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'No results cached. Click Run Backtest.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/combined/run', methods=['POST'])
def api_combined_run():
    """Start combined MQ + V3 backtest in background."""
    import threading
    task_id = f'combined_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    task_status[task_id] = {
        'status': 'running', 'progress': 0, 'message': 'Starting...',
    }

    # Get optional params from request
    params = request.get_json(silent=True) or {}

    def _run(tid):
        try:
            from services.combined_mq_v3_engine import CombinedMQV3Engine, CombinedConfig
            import json

            config = CombinedConfig(
                start_date=params.get('start_date', '2023-01-01'),
                end_date=params.get('end_date', '2025-12-31'),
                topup_stop_loss_pct=float(params.get('topup_sl_pct', 0)),
                v3_trail_pct=float(params.get('v3_trail_pct', 20.0)),
                v3_max_concurrent=int(params.get('v3_max_concurrent', 5)),
                v3_system_name=params.get('v3_system', 'PRIMARY'),
            )

            def cb(pct, msg):
                task_status[tid].update(progress=pct, message=msg)

            engine = CombinedMQV3Engine(config)
            result = engine.run(progress_callback=cb)

            # Serialize result for JSON storage
            serialized = {
                'generated_at': datetime.now().isoformat(),
                'config': {
                    'start_date': config.start_date,
                    'end_date': config.end_date,
                    'initial_capital': config.initial_capital,
                    'mq_pct': config.mq_capital_pct * 100,
                    'v3_pct': config.v3_capital_pct * 100,
                    'debt_pct': config.debt_capital_pct * 100,
                    'v3_leverage': config.v3_leverage,
                    'v3_trail_pct': config.v3_trail_pct,
                    'v3_system': config.v3_system_name,
                    'topup_sl_pct': config.topup_stop_loss_pct * 100,
                },
                'mq': {
                    'initial': result.mq_result.initial_capital,
                    'final': result.mq_result.final_value,
                    'cagr': result.mq_result.cagr,
                    'sharpe': result.mq_result.sharpe_ratio,
                    'max_dd': result.mq_result.max_drawdown,
                    'calmar': result.mq_result.calmar_ratio,
                    'trades': result.mq_result.total_trades,
                    'win_rate': result.mq_result.win_rate,
                    'topups': result.mq_result.total_topups,
                    'topup_sl_reversals': len(result.mq_result.topup_sl_log),
                    'equity_curve': result.mq_result.daily_equity,
                    'debt_curve': result.mq_result.daily_debt_fund,
                    'positions': result.mq_result.final_positions,
                    'sector_allocation': result.mq_result.sector_allocation,
                    'exit_reasons': result.mq_result.exit_reason_counts,
                },
                'v3': {
                    'initial': result.v3_initial_capital,
                    'final': result.v3_final_value,
                    'trades': result.v3_total_trades,
                    'winning': result.v3_winning_trades,
                    'win_rate': result.v3_win_rate,
                    'avg_leveraged_return': result.v3_avg_leveraged_return,
                    'profit_factor': result.v3_profit_factor,
                    'total_pnl': result.v3_total_pnl,
                    'equity_curve': result.v3_equity_curve,
                    'strategy_breakdown': result.v3_strategy_breakdown,
                    'trades_log': [
                        {
                            'symbol': t.symbol,
                            'strategy': t.strategy,
                            'entry_date': t.entry_date.strftime('%Y-%m-%d'),
                            'entry_price': t.entry_price,
                            'exit_date': t.exit_date.strftime('%Y-%m-%d'),
                            'exit_price': t.exit_price,
                            'exit_reason': t.exit_reason,
                            'base_return': t.base_return_pct,
                            'leveraged_return': t.leveraged_return_pct,
                            'margin': t.margin_deployed,
                            'pnl': t.pnl,
                        }
                        for t in result.v3_trades
                    ],
                },
                'combined': {
                    'initial': result.combined_initial,
                    'final': result.combined_final,
                    'total_return': result.combined_total_return_pct,
                    'cagr': result.combined_cagr,
                    'sharpe': result.combined_sharpe,
                    'max_dd': result.combined_max_drawdown,
                    'calmar': result.combined_calmar,
                    'equity_curve': result.combined_equity_curve,
                },
                'yearly': result.yearly_returns,
                'capital_allocation': result.capital_allocation,
            }

            # Persist to file
            results_path = Path('backtest_data/combined_mq_v3_results.json')
            results_path.parent.mkdir(exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(serialized, f, indent=2, default=str)

            task_status[tid] = {
                'status': 'complete', 'progress': 100,
                'message': 'Done', 'result': 'saved',
            }
        except Exception as e:
            logger.error(f"Combined backtest failed: {e}", exc_info=True)
            task_status[tid] = {
                'status': 'error', 'progress': 0,
                'message': str(e), 'error': str(e),
            }

    threading.Thread(target=_run, args=(task_id,), daemon=True).start()
    return jsonify({'task_id': task_id})


@app.route('/api/combined/status/<task_id>')
def api_combined_status(task_id):
    """Check combined backtest task progress."""
    status = task_status.get(task_id)
    if status:
        return jsonify(status)
    return jsonify({'error': 'Task not found'}), 404


# =============================================================================
# Model Portfolio Routes
# =============================================================================

def _calculate_xirr(cashflows, guess=0.1, max_iter=100, tol=1e-6):
    """Calculate XIRR using Newton-Raphson. cashflows = [(datetime, amount), ...]"""
    if not cashflows or len(cashflows) < 2:
        return None
    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]
    days = [(cf[0] - t0).days / 365.25 for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    rate = guess
    for _ in range(max_iter):
        npv = sum(a / (1 + rate) ** d for a, d in zip(amounts, days))
        dnpv = sum(-d * a / (1 + rate) ** (d + 1) for a, d in zip(amounts, days))
        if abs(dnpv) < 1e-12:
            return None
        new_rate = rate - npv / dnpv
        if abs(new_rate - rate) < tol:
            return new_rate
        rate = new_rate
    return None


def _serialize_model_portfolio(result, config):
    """Convert BacktestResult into JSON-serializable model portfolio data."""
    # Closed positions
    closed = []
    for t in result.trade_log:
        closed.append({
            'symbol': t.symbol, 'sector': t.sector,
            'entry_date': t.entry_date.strftime('%Y-%m-%d'),
            'entry_price': round(t.entry_price, 2),
            'exit_date': t.exit_date.strftime('%Y-%m-%d'),
            'exit_price': round(t.exit_price, 2),
            'current_price': None,
            'return_pct': round(t.return_pct * 100, 1),
            'net_pnl': round(t.net_pnl, 2),
            'exit_reason': t.exit_reason.value if hasattr(t.exit_reason, 'value') else str(t.exit_reason),
            'holding_days': t.holding_days,
            'drawdown_from_ath': None,
            'topups': t.topup_count,
            'total_invested': round(t.total_invested, 2),
            'shares': t.total_shares_at_exit,
            'status': 'closed',
        })

    # Open positions
    open_positions = []
    for p in result.final_positions:
        dd = p.get('drawdown_from_ath', 0)
        status = 'warning' if dd >= 15 else 'open'
        end_dt = datetime.strptime(config.end_date, '%Y-%m-%d')
        entry_dt = datetime.strptime(p['entry_date'], '%Y-%m-%d')
        open_positions.append({
            'symbol': p['symbol'], 'sector': p['sector'],
            'entry_date': p['entry_date'],
            'entry_price': round(p['entry_price'], 2),
            'exit_date': None, 'exit_price': None,
            'current_price': round(p['current_price'], 2),
            'return_pct': round(p['return_pct'], 1),
            'net_pnl': round(p['pnl'], 2),
            'exit_reason': None,
            'holding_days': (end_dt - entry_dt).days,
            'drawdown_from_ath': round(dd, 1),
            'topups': p.get('topups', 0),
            'total_invested': round(p.get('value', 0) - p.get('pnl', 0), 2),
            'shares': p.get('shares', 0),
            'status': status,
        })

    status_order = {'open': 0, 'warning': 1, 'closed': 2}
    positions = sorted(open_positions + closed,
                       key=lambda x: (status_order.get(x['status'], 9), -abs(x['return_pct'])))

    # XIRR
    start_dt = datetime.strptime(config.start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(config.end_date, '%Y-%m-%d')
    xirr_val = _calculate_xirr([(start_dt, -config.initial_capital), (end_dt, result.final_value)])

    summary = {
        'total_positions': len(positions),
        'open_count': len(open_positions),
        'closed_count': len(closed),
        'warning_count': sum(1 for p in open_positions if p['status'] == 'warning'),
        'xirr': round(xirr_val * 100, 2) if xirr_val else None,
        'cagr': round(result.cagr, 2),
        'sharpe': round(result.sharpe_ratio, 2),
        'sortino': round(result.sortino_ratio, 2),
        'max_drawdown': round(result.max_drawdown, 2),
        'calmar': round(result.calmar_ratio, 2),
        'final_value': round(result.final_value, 2),
        'initial_capital': config.initial_capital,
        'total_return_pct': round(result.total_return_pct, 2),
        'total_topups': result.total_topups,
        'total_trades': result.total_trades,
        'win_rate': round(result.win_rate, 1),
    }

    eq_dates = sorted(result.daily_equity.keys())
    equity_curve = {d: round(result.daily_equity[d], 2)
                    for i, d in enumerate(eq_dates) if i % 5 == 0 or i == len(eq_dates) - 1}

    return {
        'generated_at': datetime.now().isoformat(),
        'config': {'start_date': config.start_date, 'end_date': config.end_date,
                   'initial_capital': config.initial_capital, 'portfolio_size': config.portfolio_size},
        'positions': positions,
        'summary': summary,
        'equity_curve': equity_curve,
        'sector_allocation': result.sector_allocation,
        'exit_reason_counts': result.exit_reason_counts,
    }


@app.route('/model-portfolio')
def model_portfolio_dashboard():
    """Model Portfolio Dashboard - MQ strategy positions from Jan 2023."""
    return render_template(
        'model_portfolio.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/api/model-portfolio')
def api_model_portfolio():
    """Return cached model portfolio results."""
    results_path = Path('backtest_data/model_portfolio_results.json')
    if results_path.exists():
        return jsonify(json.loads(results_path.read_text(encoding='utf-8')))
    return jsonify({'error': 'No results cached. Click Run Backtest.'}), 404


@app.route('/api/model-portfolio/run', methods=['POST'])
def api_model_portfolio_run():
    """Start model portfolio backtest in background thread."""
    import threading
    task_id = f'model_portfolio_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    task_status[task_id] = {'status': 'running', 'progress': 0, 'message': 'Starting...'}

    def _run(tid):
        try:
            from services.mq_backtest_engine import MQBacktestEngine
            from services.mq_portfolio import MQBacktestConfig

            task_status[tid].update(progress=5, message='Loading universe data...')
            config = MQBacktestConfig(
                start_date='2023-01-01', end_date='2026-02-17',
                initial_capital=10_000_000, portfolio_size=20,
                equity_allocation_pct=0.95, hard_stop_loss=0.50,
                rebalance_ath_drawdown=0.20,
            )

            def cb(day_idx, total_days, current_date, msg):
                pct = (day_idx / total_days * 100) if total_days else 0
                task_status[tid].update(progress=round(pct, 1), message=msg)

            engine = MQBacktestEngine(config)
            result = engine.run(progress_callback=cb)
            serialized = _serialize_model_portfolio(result, config)

            results_path = Path('backtest_data/model_portfolio_results.json')
            results_path.parent.mkdir(exist_ok=True)
            results_path.write_text(json.dumps(serialized, indent=2, default=str), encoding='utf-8')

            task_status[tid] = {'status': 'completed', 'progress': 100, 'message': 'Done'}
        except Exception as e:
            logger.error(f"Model portfolio backtest failed: {e}", exc_info=True)
            task_status[tid] = {'status': 'error', 'progress': 0, 'message': str(e)}

    threading.Thread(target=_run, args=(task_id,), daemon=True).start()
    return jsonify({'task_id': task_id})


@app.route('/api/model-portfolio/status/<task_id>')
def api_model_portfolio_status(task_id):
    """Check model portfolio task progress."""
    status = task_status.get(task_id)
    if status:
        return jsonify(status)
    return jsonify({'error': 'Task not found'}), 404


# =============================================================================
# Multi-Strategy Dashboard
# =============================================================================

@app.route('/strategies')
def strategies_dashboard():
    """Unified multi-strategy dashboard."""
    return render_template(
        'strategies_dashboard.html',
        authenticated=is_authenticated(),
        user_name=session.get('user_name', 'User'),
    )


@app.route('/api/strategies/nondirectional/summary')
def api_nondirectional_summary():
    """Get BankNifty Non-Directional V3 backtest summary (best config)."""
    try:
        import csv as csv_mod
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'nondirectional_v3_summary.csv')
        if not os.path.exists(csv_path):
            return jsonify({'error': 'No V3 summary CSV found'}), 404

        best_label = 'BANKNIFTY_short_strangle_biweekly_monthly_BB_only_SD1.5_TF2.0_L5'
        best = None
        all_rows = []

        with open(csv_path) as f:
            for row in csv_mod.DictReader(f):
                all_rows.append(row)
                if row['label'] == best_label:
                    best = row

        if not best and all_rows:
            # Fallback: pick highest CAGR
            best = max(all_rows, key=lambda r: float(r.get('cagr_pct', 0)))

        if not best:
            return jsonify({'error': 'No data'}), 404

        # Convert numeric fields
        for k in ['total_trades', 'max_consecutive_losses', 'lots']:
            if k in best:
                best[k] = int(float(best[k]))
        for k in ['win_rate', 'total_pnl', 'total_return_pct', 'cagr_pct',
                   'profit_factor', 'max_drawdown_pct', 'avg_premium',
                   'avg_zone_width_pct', 'zone_hold_rate', 'avg_pnl_per_trade',
                   'avg_win', 'avg_loss', 'max_win', 'max_loss', 'final_capital']:
            if k in best:
                best[k] = round(float(best[k]), 2)

        return jsonify(best)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategies/nondirectional/trades')
def api_nondirectional_trades():
    """Get BankNifty Non-Directional V3 trade log (best config only)."""
    try:
        import csv as csv_mod
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'nondirectional_v3_trades.csv')
        if not os.path.exists(csv_path):
            return jsonify([])

        best_label = 'BANKNIFTY_short_strangle_biweekly_monthly_BB_only_SD1.5_TF2.0_L5'
        trades = []

        with open(csv_path) as f:
            for row in csv_mod.DictReader(f):
                if row.get('config') == best_label:
                    # Convert numeric fields
                    for k in ['entry_price', 'call_strike', 'put_strike',
                              'zone_width_pct', 'premium', 'exit_price',
                              'gross_pnl', 'net_pnl', 'pnl_rs', 'iv', 'atr']:
                        if k in row and row[k]:
                            try:
                                row[k] = round(float(row[k]), 2)
                            except ValueError:
                                pass
                    trades.append(row)

        return jsonify(trades)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategies/nondirectional/equity-curve')
def api_nondirectional_equity_curve():
    """Get BankNifty Non-Directional equity curve from trade log."""
    try:
        import csv as csv_mod
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'nondirectional_v3_trades.csv')
        if not os.path.exists(csv_path):
            return jsonify([])

        best_label = 'BANKNIFTY_short_strangle_biweekly_monthly_BB_only_SD1.5_TF2.0_L5'
        capital = 10_00_000  # 10L starting capital
        curve = [{'date': '2023-04-01', 'capital': capital, 'pnl': 0}]

        with open(csv_path) as f:
            for row in csv_mod.DictReader(f):
                if row.get('config') == best_label:
                    pnl_rs = float(row.get('pnl_rs', 0))
                    capital += pnl_rs
                    curve.append({
                        'date': row.get('exit_date', '')[:10],
                        'capital': round(capital, 2),
                        'pnl': round(pnl_rs, 2),
                    })

        return jsonify(curve)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Trident — PA_MACD + RangeBreakout Live Trading
# =============================================================================

TRIDENT_CONFIG = {
    'max_positions': 20,
    'position_size_pct': 0.05,
    'max_daily_orders': 20,
    'max_daily_loss_pct': 3.0,
    'capital': 10_000_000,
    'paper_trading_mode': True,
    'live_trading_enabled': False,
    'enabled': True,
}


@app.route('/trident')
def trident_dashboard():
    """Trident execution dashboard."""
    return render_template('trident_dashboard.html')


@app.route('/api/trident/state')
def api_trident_state():
    """Get full Trident system state."""
    try:
        from services.trident_executor import get_trident_executor
        executor = get_trident_executor(TRIDENT_CONFIG)
        return jsonify(executor.get_state())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trident/scan', methods=['POST'])
def api_trident_scan():
    """Trigger a manual Trident scan."""
    import threading

    def _run_scan():
        try:
            from services.trident_executor import get_trident_executor
            executor = get_trident_executor(TRIDENT_CONFIG)
            executor.run_scan()
        except Exception as e:
            logger.error(f"[Trident] Manual scan error: {e}")

    t = threading.Thread(target=_run_scan, daemon=True)
    t.start()
    return jsonify({'status': 'started', 'message': 'Scan running in background'})


@app.route('/api/trident/kill-switch', methods=['POST'])
def api_trident_kill_switch():
    """Emergency exit all Trident positions."""
    try:
        from services.trident_executor import get_trident_executor
        executor = get_trident_executor(TRIDENT_CONFIG)
        results = executor.emergency_exit_all()
        return jsonify({'status': 'done', 'closed': len(results), 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trident/toggle-mode', methods=['POST'])
def api_trident_toggle_mode():
    """Toggle Trident paper/live mode."""
    is_paper = TRIDENT_CONFIG.get('paper_trading_mode', True)
    TRIDENT_CONFIG['paper_trading_mode'] = not is_paper
    TRIDENT_CONFIG['live_trading_enabled'] = is_paper
    mode = 'PAPER' if TRIDENT_CONFIG['paper_trading_mode'] else 'LIVE'
    logger.info(f"[Trident] Mode toggled to {mode}")
    return jsonify({'mode': mode})


@app.route('/api/trident/toggle-enabled', methods=['POST'])
def api_trident_toggle_enabled():
    """Enable/disable Trident system."""
    TRIDENT_CONFIG['enabled'] = not TRIDENT_CONFIG.get('enabled', True)
    state = 'ENABLED' if TRIDENT_CONFIG['enabled'] else 'DISABLED'
    logger.info(f"[Trident] System {state}")
    return jsonify({'enabled': TRIDENT_CONFIG['enabled']})


@app.route('/api/trident/trades')
def api_trident_trades():
    """Get Trident trade history."""
    try:
        from services.trident_db import get_trident_db
        db = get_trident_db()
        limit = request.args.get('limit', 50, type=int)
        return jsonify(db.get_trades(limit))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trident/equity-curve')
def api_trident_equity_curve():
    """Get Trident equity curve."""
    try:
        from services.trident_db import get_trident_db
        db = get_trident_db()
        return jsonify(db.get_equity_curve())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trident/pending-signals')
def api_trident_pending_signals():
    """Get pending stop signals."""
    try:
        from services.trident_db import get_trident_db
        db = get_trident_db()
        return jsonify(db.get_pending_signals())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- Trident Scheduled Jobs ----

def _trident_check_exits():
    """3:15 PM — Check SL/TP/MaxHold for active positions."""
    if not TRIDENT_CONFIG.get('enabled', True):
        return
    try:
        from services.trident_executor import get_trident_executor
        from services.trident_scanner import load_daily_data_from_db, check_exits
        executor = get_trident_executor(TRIDENT_CONFIG)
        db_path = str(DATA_DIR / 'market_data.db')
        from services.trident_executor import TRIDENT_UNIVERSE
        symbol_data = load_daily_data_from_db(TRIDENT_UNIVERSE, db_path)
        positions = executor.db.get_active_positions()
        exit_signals = check_exits(positions, symbol_data, TRIDENT_CONFIG)
        for ex in exit_signals:
            executor.execute_exit(ex)
        logger.info(f"[Trident] Exit check: {len(exit_signals)} exits")
    except Exception as e:
        logger.error(f"[Trident] Exit check error: {e}")


def _trident_full_scan():
    """3:20 PM — Full scan: check pending triggers + new signals."""
    if not TRIDENT_CONFIG.get('enabled', True):
        return
    try:
        from services.trident_executor import get_trident_executor
        executor = get_trident_executor(TRIDENT_CONFIG)
        result = executor.run_scan()
        logger.info(f"[Trident] Full scan done: {result.get('pamacd_signals', 0)} PA_MACD, "
                    f"{result.get('rb_signals', 0)} RB signals")
    except Exception as e:
        logger.error(f"[Trident] Full scan error: {e}")


try:
    scheduler.add_job(
        _trident_check_exits,
        'cron', day_of_week='mon-fri', hour=15, minute=15,
        id='trident_check_exits', replace_existing=True,
    )
    scheduler.add_job(
        _trident_full_scan,
        'cron', day_of_week='mon-fri', hour=15, minute=20,
        id='trident_full_scan', replace_existing=True,
    )
    logger.info("Trident scheduled jobs registered: exits(15:15), scan(15:20)")
except Exception as e:
    logger.warning(f"Could not register Trident scheduled jobs: {e}")


# =============================================================================
# NWV - Nifty Weekly View (Phase 0: view-only, no orders)
# =============================================================================

def _nwv_build_weekly_state():
    """Sunday 22:00 IST: compute this-coming-week's CPR + pivots from last
    week's NIFTY HLC. Also refresh monthly CPR when entering a new month.
    Runs before Monday morning so the weekly state is pre-populated.
    """
    try:
        from datetime import date as _d, timedelta as _td
        from services.nwv_engine import get_nwv_engine
        from services.nwv_data import fetch_weekly_hlc, fetch_monthly_hlc, fetch_monday_open

        today = _d.today()
        # Next Monday from today (or today if it's a Monday evening run).
        days_until_mon = (7 - today.weekday()) % 7
        if days_until_mon == 0 and today.weekday() == 0:
            week_start = today
        else:
            week_start = today + _td(days=days_until_mon or 1)

        weekly = fetch_weekly_hlc(week_start)
        if not weekly:
            logger.warning("[NWV] weekly HLC unavailable - skip state build")
            return

        monthly = fetch_monthly_hlc(week_start)
        monthly_cpr = None
        if monthly:
            from services.nwv_engine import compute_cpr, compute_pivots
            m = compute_cpr(monthly['high'], monthly['low'], monthly['close'])
            monthly_cpr = {'tc': m['tc'], 'bc': m['bc'], 'pp': m['pp']}

        engine = get_nwv_engine()
        row = engine.build_weekly_state(
            week_start=week_start,
            prev_high=weekly['high'], prev_low=weekly['low'],
            prev_close=weekly['close'],
            prev_fri_close=weekly['prev_fri_close'],
            spot_ref=weekly['close'],
            monthly=monthly_cpr,
            notes='auto-built Sunday 22:00 job',
        )
        logger.info(
            f"[NWV] weekly_state built for {week_start}: bucket={row['cpr_bucket']} "
            f"width={row['cpr_width_pct']:.3f}% PP={row['pivot_pp']:.1f} "
            f"S1={row['pivot_s1']:.1f} R1={row['pivot_r1']:.1f}"
        )
    except Exception as e:
        logger.error(f"[NWV] weekly_state build failed: {e}", exc_info=True)


def _nwv_compute_view():
    """Monday 09:46 IST: after the first 30-min candle closes, compute
    the view and fire the WhatsApp notification. No Kite orders.
    """
    try:
        from datetime import date as _d
        from services.nwv_engine import get_nwv_engine
        from services.nwv_data import (
            fetch_first_30min_candle, fetch_monday_open,
            fetch_vix_and_percentile, compute_adx_daily, fetch_daily_pivots,
        )

        today = _d.today()
        # Find this week's Monday
        week_start = today - timedelta(days=today.weekday())

        engine = get_nwv_engine()
        weekly_state = engine.db.get_weekly_state(week_start)
        if not weekly_state:
            # Best-effort on-demand build for today
            _nwv_build_weekly_state()
            weekly_state = engine.db.get_weekly_state(week_start)
            if not weekly_state:
                logger.error("[NWV] no weekly_state available - cannot compute view")
                return

        first_candle = fetch_first_30min_candle()
        if not first_candle:
            logger.warning("[NWV] first 30-min candle unavailable - skip")
            return

        mon_open = first_candle['open']  # 09:15 open is the candle open
        prev_fri_close = weekly_state.get('prev_fri_close') or weekly_state.get('prev_week_close')

        vix_val, vix_rank = fetch_vix_and_percentile()
        adx_d = compute_adx_daily()
        daily_pivots = fetch_daily_pivots()

        view = engine.compute_view(
            week_start=week_start,
            weekly_state=weekly_state,
            mon_open=mon_open,
            prev_fri_close=prev_fri_close,
            first_candle=first_candle,
            daily_pivots=daily_pivots,
            vix_value=vix_val,
            vix_pct_rank=vix_rank,
            adx_daily=adx_d,
            spot=first_candle['close'],
        )

        logger.info(
            f"[NWV] {week_start} FINAL VIEW={view['final_view']} "
            f"conviction={view['conviction']}/5 instrument={view['instrument_choice']} "
            f"gap={view['gap_pct']:.2f}% cpr_bucket={view['cpr_bucket']}"
        )

        # WhatsApp / email notification
        try:
            from services.notifications import get_notification_service
            ns = get_notification_service({
                'whatsapp_enabled': True, 'email_enabled': True,
                'twilio_account_sid': os.getenv('TWILIO_ACCOUNT_SID', ''),
                'twilio_auth_token': os.getenv('TWILIO_AUTH_TOKEN', ''),
                'twilio_whatsapp_from': os.getenv('TWILIO_FROM_WHATSAPP', 'whatsapp:+14155238886'),
                'twilio_whatsapp_to': os.getenv('TWILIO_TO_WHATSAPP', ''),
                'email_from': os.getenv('EMAIL_FROM', 'arun.castromin@gmail.com'),
                'email_to': os.getenv('EMAIL_TO', 'arun.castromin@gmail.com'),
                'email_app_password': os.getenv('GMAIL_APP_PASSWORD', ''),
                'smtp_host': 'smtp.gmail.com', 'smtp_port': 587,
            })
            # Compact ≤ 500-char message
            exp_lo = view.get('expected_range_low')
            exp_hi = view.get('expected_range_high')
            if exp_lo is not None and exp_hi is not None:
                rng = f"{exp_lo:.0f}-{exp_hi:.0f}"
            elif exp_lo is not None:
                rng = f">{exp_lo:.0f}"
            elif exp_hi is not None:
                rng = f"<{exp_hi:.0f}"
            else:
                rng = "-"
            body = (
                f"NIFTY Weekly View - {week_start}\n"
                f"View: {view['final_view']}  conviction {view['conviction']}/5\n"
                f"CPR {view['cpr_bucket']} {view.get('cpr_width_pct', 0):.2f}%\n"
                f"1st 30m: {view['first_candle_body']}, {view['first_candle_pos']}\n"
                f"Gap: {view['gap_pct']:+.2f}% ({view['gap_tier']})\n"
                f"VIX: {view.get('vix_value') or '-'} ({view.get('vix_pct_rank') or '-'}%ile)  "
                f"ADX: {view.get('adx_daily') or '-'}\n"
                f"Trade: {view['instrument_choice']}  Range: {rng}\n"
                f"Time stop: {view['time_stop']}\n"
                f"http://94.136.185.54:5000/app/nwv"
            )
            ns.send_alert('system', 'NWV Monday View', body, data=None, priority='high')
        except Exception as e:
            logger.warning(f"[NWV] notification dispatch failed: {e}")
    except Exception as e:
        logger.error(f"[NWV] view compute failed: {e}", exc_info=True)


try:
    scheduler.add_job(
        _nwv_build_weekly_state,
        'cron', day_of_week='sun', hour=22, minute=0,
        id='nwv_weekly_state', replace_existing=True,
    )
    scheduler.add_job(
        _nwv_compute_view,
        'cron', day_of_week='mon', hour=9, minute=46,
        id='nwv_compute_view', replace_existing=True, max_instances=1,
    )
    logger.info("NWV scheduled jobs registered: weekly_state(Sun 22:00), compute_view(Mon 09:46)")
except Exception as e:
    logger.warning(f"Could not register NWV scheduled jobs: {e}")


# ─── NWV API endpoints ─────────────────────────────────────

@app.route('/api/nwv/view')
def api_nwv_view():
    """Latest NWV view row (joined with its weekly_state)."""
    try:
        from services.nwv_engine import get_nwv_engine
        import json as _json
        engine = get_nwv_engine()
        v = engine.db.latest_view()
        if not v:
            return jsonify({'view': None, 'weekly_state': None,
                             'message': 'No view computed yet. Next fire: Monday 09:46 IST.'})
        ws = engine.db.get_weekly_state(v.get('week_start'))
        # Deserialize JSON text columns for convenience
        for k in ('stacked_supports', 'stacked_resistances'):
            if v.get(k):
                try:
                    v[k] = _json.loads(v[k])
                except Exception:
                    pass
        return jsonify({'view': v, 'weekly_state': ws})
    except Exception as e:
        logger.error(f"[API] /api/nwv/view error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/nwv/weekly-state')
def api_nwv_weekly_state():
    """Current week's CPR + pivots row."""
    try:
        from services.nwv_engine import get_nwv_engine
        ws = get_nwv_engine().db.latest_weekly_state()
        return jsonify({'weekly_state': ws})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nwv/views-history')
def api_nwv_views_history():
    """Recent NWV views — up to `n` entries."""
    try:
        from services.nwv_engine import get_nwv_engine
        import json as _json
        n = int(request.args.get('n', 20))
        rows = get_nwv_engine().db.recent_views(n=n)
        for r in rows:
            for k in ('stacked_supports', 'stacked_resistances'):
                if r.get(k):
                    try:
                        r[k] = _json.loads(r[k])
                    except Exception:
                        pass
        return jsonify({'views': rows})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nwv/recompute', methods=['POST'])
def api_nwv_recompute():
    """Manual trigger — useful for re-testing after data updates or when
    Monday morning has already passed and we want to see the view."""
    try:
        _nwv_compute_view()
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error='Internal server error'), 500


# =============================================================================
# EOD Breakout Scanner — daily-bar paper trading service
# =============================================================================

@app.route('/api/eod/state')
def api_eod_state():
    """Combined state for the EOD breakout multi-system page."""
    try:
        from services.eod_breakout_db import get_eod_breakout_db, ALL_SYSTEMS
        from services.eod_breakout_scanner import SYSTEM_CONFIG, UNIVERSE_LOADERS
        db = get_eod_breakout_db()
        systems = []
        for sys_id in ALL_SYSTEMS:
            cfg = SYSTEM_CONFIG[sys_id]
            stats = db.get_stats(sys_id)
            try:
                universe_size = len(UNIVERSE_LOADERS[sys_id]())
            except Exception:
                universe_size = 0
            today_iso = datetime.now().date().isoformat()
            today_state = db.get_daily_state(sys_id, today_iso) or {}
            systems.append({
                'system_id': sys_id,
                'description': cfg['description'],
                'capital': cfg['capital'],
                'risk_per_trade_pct': cfg['risk_per_trade_pct'],
                'max_concurrent': cfg['max_concurrent'],
                'cost_pct': cfg['cost_pct'],
                'vol_threshold_mult': cfg['vol_threshold_mult'],
                'target_pct': cfg['target_pct'],
                'initial_hard_stop_pct': cfg['initial_hard_stop_pct'],
                'universe_size': universe_size,
                'open_positions': stats.get('open_positions', 0),
                'total_trades': stats.get('total_trades', 0),
                'win_rate': stats.get('win_rate', 0),
                'profit_factor': stats.get('profit_factor', 0),
                'total_pnl': round(stats.get('total_pnl', 0) or 0, 2),
                'avg_days_held': round(stats.get('avg_days_held', 0) or 0, 1),
                'today_signals': today_state.get('signals_generated', 0),
                'today_fills': today_state.get('fills_today', 0),
                'today_exits': today_state.get('exits_today', 0),
            })
        return jsonify({'systems': systems})
    except Exception as e:
        logger.error(f"[EOD] state error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/eod/<system_id>/positions')
def api_eod_positions(system_id):
    try:
        from services.eod_breakout_db import get_eod_breakout_db
        db = get_eod_breakout_db()
        return jsonify({'positions': db.get_open_positions(system_id)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eod/<system_id>/signals')
def api_eod_signals(system_id):
    try:
        from services.eod_breakout_db import get_eod_breakout_db
        db = get_eod_breakout_db()
        return jsonify({'signals': db.get_recent_signals(system_id, limit=100)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eod/<system_id>/trades')
def api_eod_trades(system_id):
    try:
        from services.eod_breakout_db import get_eod_breakout_db
        db = get_eod_breakout_db()
        return jsonify({'trades': db.get_recent_trades(system_id, limit=200)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eod/<system_id>/equity-curve')
def api_eod_equity_curve(system_id):
    try:
        from services.eod_breakout_db import get_eod_breakout_db
        db = get_eod_breakout_db()
        return jsonify({'curve': db.get_equity_curve(system_id)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eod/<action>', methods=['POST'])
def api_eod_manual(action):
    """Manual triggers — useful for testing or catch-up runs."""
    try:
        from services.eod_breakout_scanner import scan_eod, record_morning_fills, check_exits, run_full_daily_cycle
        if action == 'scan':
            return jsonify(scan_eod())
        elif action == 'fill':
            return jsonify(record_morning_fills())
        elif action == 'exit-check':
            return jsonify(check_exits())
        elif action == 'full-cycle':
            return jsonify(run_full_daily_cycle())
        else:
            return jsonify({'error': f'unknown action: {action}'}), 400
    except Exception as e:
        logger.error(f"[EOD-MANUAL] {action} error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Scheduled EOD breakout jobs
def _eod_breakout_scan_job():
    """16:00 IST Mon-Fri — generate breakout signals from today's daily bars."""
    try:
        from services.eod_breakout_scanner import scan_eod
        result = scan_eod()
        logger.info(f"[EOD-BREAKOUT] daily scan: {result}")
    except Exception as e:
        logger.error(f"[EOD-BREAKOUT] scan job error: {e}", exc_info=True)


def _eod_breakout_fill_job():
    """09:20 IST Mon-Fri — fill PENDING signals at today's open."""
    try:
        from services.eod_breakout_scanner import record_morning_fills
        result = record_morning_fills()
        logger.info(f"[EOD-BREAKOUT] morning fills: {result}")
    except Exception as e:
        logger.error(f"[EOD-BREAKOUT] fill job error: {e}", exc_info=True)


def _eod_breakout_exit_job():
    """16:05 IST Mon-Fri — check open positions for target/stop/max-hold."""
    try:
        from services.eod_breakout_scanner import check_exits
        result = check_exits()
        logger.info(f"[EOD-BREAKOUT] exit check: {result}")
    except Exception as e:
        logger.error(f"[EOD-BREAKOUT] exit job error: {e}", exc_info=True)


try:
    scheduler.add_job(
        _eod_breakout_fill_job,
        'cron', day_of_week='mon-fri', hour=9, minute=20,
        id='eod_breakout_fill', replace_existing=True,
    )
    scheduler.add_job(
        _eod_breakout_scan_job,
        'cron', day_of_week='mon-fri', hour=16, minute=0,
        id='eod_breakout_scan', replace_existing=True,
    )
    scheduler.add_job(
        _eod_breakout_exit_job,
        'cron', day_of_week='mon-fri', hour=16, minute=5,
        id='eod_breakout_exit', replace_existing=True,
    )
    logger.info("EOD Breakout scheduled jobs registered: fill (9:20), scan (16:00), exit-check (16:05)")
except Exception as e:
    logger.warning(f"Could not register EOD Breakout scheduled jobs: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Ensure directories exist
    templates_dir = Path(__file__).parent / 'templates'
    static_dir = Path(__file__).parent / 'static'
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)

    # Restore Maruthi trading mode from DB (survives Flask restarts)
    try:
        from services.maruthi_db import get_maruthi_db
        _mdb = get_maruthi_db()
        _saved_mode = _mdb.get_setting('trading_mode', 'PAPER')
        if _saved_mode == 'LIVE':
            MARUTHI_DEFAULTS['paper_trading_mode'] = False
            MARUTHI_DEFAULTS['live_trading_enabled'] = True
            logger.info("[Maruthi] Restored LIVE trading mode from DB")
        else:
            logger.info("[Maruthi] Mode is PAPER (default)")
        _saved_enabled = _mdb.get_setting('enabled', 'True')
        if _saved_enabled == 'False':
            MARUTHI_DEFAULTS['enabled'] = False
            logger.info("[Maruthi] Restored DISABLED state from DB")
    except Exception as e:
        logger.warning(f"[Maruthi] Could not restore mode from DB: {e}")

    # Auto-start Maruthi ticker if strategy is enabled
    if MARUTHI_DEFAULTS.get('enabled', True):
        try:
            from services.maruthi_ticker import get_maruthi_ticker
            ticker = get_maruthi_ticker(MARUTHI_DEFAULTS)
            if not ticker.is_connected:
                ticker.start()
                logger.info("[Maruthi] Ticker auto-started on Flask boot")
        except Exception as e:
            logger.warning(f"[Maruthi] Ticker auto-start failed on boot: {e}")

    # Auto-start NAS ticker if enabled and during market hours
    if NAS_DEFAULTS.get('enabled', True):
        _now = datetime.now()
        _market_open = _now.replace(hour=9, minute=15, second=0)
        _market_close = _now.replace(hour=15, minute=30, second=0)
        if _market_open <= _now <= _market_close and _now.weekday() < 5:
            try:
                from services.nas_ticker import get_nas_ticker
                ticker = get_nas_ticker(NAS_DEFAULTS)
                if not ticker.is_running:
                    ticker.start()
                    logger.info("[NAS] Ticker auto-started on Flask boot")
            except Exception as e:
                logger.warning(f"[NAS] Ticker auto-start failed on boot: {e}")
        else:
            logger.info("[NAS] Outside market hours, ticker will start at 9:16 via cron")

    # BNF: Run initial scan to populate BB state from daily bars
    if BNF_DEFAULTS.get('enabled', True):
        try:
            from services.bnf_executor import get_bnf_executor
            _bnf_exec = get_bnf_executor(BNF_DEFAULTS)
            _bnf_scan = _bnf_exec.scanner.scan()
            if _bnf_scan and not _bnf_scan.get('error'):
                _bnf_exec.db.update_state(
                    bb_state=_bnf_scan['bb_state'],
                    squeeze_count=_bnf_scan['squeeze_count'],
                    bb_width=_bnf_scan['bb_width'],
                    bb_width_ma=_bnf_scan['bb_width_ma'],
                    sma_value=_bnf_scan['sma'],
                    atr_value=_bnf_scan['atr'],
                    direction=_bnf_scan['direction'],
                    trend_strength=_bnf_scan['trend_strength'],
                    last_close=_bnf_scan['spot'],
                    last_scan_time=datetime.now().isoformat(),
                )
                logger.info(f"[BNF] Initial scan: {_bnf_scan['bb_state']}, "
                           f"spot={_bnf_scan['spot']}, dir={_bnf_scan['direction']}, "
                           f"squeeze={_bnf_scan['squeeze_count']}")
            else:
                logger.warning(f"[BNF] Initial scan failed: {_bnf_scan.get('error', 'unknown')}")
        except Exception as e:
            logger.warning(f"[BNF] Initial scan error on boot: {e}")

    # Run app with SocketIO support
    socketio.run(
        app,
        host='127.0.0.1',
        port=5000,
        debug=os.getenv('FLASK_DEBUG', '0') == '1',
        allow_unsafe_werkzeug=True  # Required for development mode
    )
