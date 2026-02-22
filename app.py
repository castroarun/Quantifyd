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
from datetime import datetime, timedelta
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
    STRIKE_METHODS, EXIT_RULES, RISK_FREE_RATE,
    MQ_DEFAULTS, KC6_DEFAULTS,
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
        return redirect(url_for('backtest_page'))

    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        flash(f'Login failed: {str(e)}', 'error')
        return redirect(url_for('index'))


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
@login_required
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
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error='Internal server error'), 500


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Ensure directories exist
    templates_dir = Path(__file__).parent / 'templates'
    static_dir = Path(__file__).parent / 'static'
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)

    # Run app with SocketIO support
    socketio.run(
        app,
        host='127.0.0.1',
        port=5000,
        debug=os.getenv('FLASK_DEBUG', '0') == '1',
        allow_unsafe_werkzeug=True  # Required for development mode
    )
