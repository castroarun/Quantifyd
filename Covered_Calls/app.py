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
    STRIKE_METHODS, EXIT_RULES, RISK_FREE_RATE
)

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

# Background scheduler for async tasks
scheduler = BackgroundScheduler()
scheduler.start()

# In-memory task status storage
task_status = {}


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
        prices = get_historical_prices(symbol, period=period)
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

    # Run app
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=os.getenv('FLASK_DEBUG', '0') == '1'
    )
