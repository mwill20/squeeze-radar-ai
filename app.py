import gradio as gr
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
# Removed finvizfinance dependency due to compatibility issues
# Using static ticker list for MVP

def get_stock_analysis(ticker):
    if not ticker:
        return "Please enter a stock ticker.", None

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # --- Data Fetching and Validation ---
        if info.get('regularMarketPrice') is None:
             return f"Could not find data for ticker '{ticker}'. Please check the symbol.", None

        # Fetch historical data for the last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return f"No historical data found for {ticker}.", None

        # --- Technical Indicator Calculation ---
        hist.ta.rsi(length=14, append=True)
        hist.ta.bbands(length=20, append=True)
        hist.reset_index(inplace=True)

        # --- Key Metrics Formatting ---
        price = info.get('currentPrice', 'N/A')
        short_float = info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else 'N/A'
        short_ratio = info.get('shortRatio', 'N/A')
        volume = info.get('volume', 'N/A')
        latest_rsi = hist['RSI_14'].iloc[-1] if not hist.empty else 'N/A'
        
        # Calculate Volume Spike
        avg_volume_20d = hist['Volume'].tail(20).mean()
        volume_spike = (volume / avg_volume_20d) if avg_volume_20d > 0 else 0

        # --- News Fetching ---
        news = stock.news
        news_list = []
        if news:
            for article in news[:3]:  # Show top 3 articles
                title = article.get('title')
                link = article.get('link', '#')
                publisher = article.get('publisher', 'Unknown')
                if title and link:
                    news_list.append(f'<li><a href="{link}" target="_blank">{title}</a> <span style="color:gray;">- {publisher}</span></li>')
        news_html = '<ul>' + ''.join(news_list) + '</ul>' if news_list else '<span>No recent news found.</span>'

        # --- Key Metrics as DataFrame ---
        metrics_data = [
            ["Current Price", f"{price:.2f}", "The most recent trading price."],
            ["Volume", f"{volume:,} ({volume_spike:.1f}x avg)", "Today's volume vs 20-day average."],
            ["RSI (14-day)", f"{latest_rsi:.2f}", "Relative Strength Index. >70 is overbought, <30 is oversold."],
            ["Short % of Float", f"{short_float:.2f}%", "Percentage of freely traded shares that are currently sold short."],
            ["Short Ratio", f"{short_ratio}", "Days required for short sellers to cover their positions."]
        ]
        metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value", "Description"])

        # --- Plotly Chart Generation with Subplots for MACD ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.1, row_heights=[0.7, 0.3])

        # Candlestick chart on the first row
        fig.add_trace(go.Candlestick(x=hist['Date'], open=hist['Open'], high=hist['High'],
                                   low=hist['Low'], close=hist['Close'], name='Price'),
                    row=1, col=1)

        # Bollinger Bands on the first row
        fig.add_trace(go.Scatter(x=hist['Date'], y=hist['BBU_20_2.0'], line=dict(color='rgba(135, 206, 250, 0.5)'), name='Upper Band'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist['Date'], y=hist['BBL_20_2.0'], line=dict(color='rgba(135, 206, 250, 0.5)'), fill='tonexty', fillcolor='rgba(135, 206, 250, 0.1)', name='Lower Band'), row=1, col=1)

        # MACD calculation and plot on the second row
        hist.ta.macd(append=True)
        fig.add_trace(go.Scatter(x=hist['Date'], y=hist['MACD_12_26_9'], line=dict(color='blue'), name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=hist['Date'], y=hist['MACDs_12_26_9'], line=dict(color='orange'), name='Signal'), row=2, col=1)
        fig.add_trace(go.Bar(x=hist['Date'], y=hist['MACDh_12_26_9'], name='Histogram', marker_color=np.where(hist['MACDh_12_26_9'] < 0, 'red', 'green')), row=2, col=1)

        fig.update_layout(
            title_text=f"{info.get('shortName', ticker.upper())} - Price & Momentum",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)

        return (metrics_df, news_html), fig

    except Exception as e:
        return f"An error occurred: {e}", None

def run_squeeze_scanner():
    """
    Scans a predefined list of stocks for squeeze candidates
    and enriches with yfinance data.
    """
    try:
        # For the MVP, we'll use a predefined list of tickers known for high volatility and short interest
        candidate_tickers = [
            'GME', 'AMC', 'BBBY', 'BYND', 'UPST', 'CVNA', 'AI', 'PLTR',
            'SOFI', 'RIVN', 'LCID', 'NKLA', 'W', 'MARA', 'RIOT', 'SPCE',
            'TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'META', 'GOOGL', 'NFLX'
        ]
        # Remove known problematic tickers
        if 'NKLA' in candidate_tickers:  # NKLA has known 404 issues
            candidate_tickers.remove('NKLA')
        if 'BBBYQ' in candidate_tickers:  # Use BBBY instead of BBBYQ
            candidate_tickers.remove('BBBYQ')
        
        # In a future version, we would implement dynamic screening with a library like finvizfinance
        # to get stocks with high short interest in real-time

    except Exception as e:
        # Handle any unexpected errors
        error_df = pd.DataFrame()
        error_df['Error'] = [f"An unexpected error occurred: {e}"]
        return error_df

    scan_results = []
    
    # Step 2: Analyze the filtered list with yfinance for more detail
    for ticker_symbol in gr.Progress(track_tqdm=True).tqdm(candidate_tickers, desc="Analyzing Top Candidates"):
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            
            # Skip if we can't get basic info
            if not info or 'currentPrice' not in info:
                continue
                
            # Get historical data for calculations
            hist = stock.history(period="1mo")
            if hist.empty:
                continue
                
            # Calculate technical indicators
            hist.ta.rsi(length=14, append=True)
            rsi = hist['RSI_14'].iloc[-1] if 'RSI_14' in hist.columns else 50  # Default to neutral RSI if calculation fails
            
            # Get key metrics
            short_float = info.get('shortPercentOfFloat', 0)
            price = info.get('currentPrice', 0)
            volume = info.get('volume', 0)
            avg_volume = hist['Volume'].mean()
            volume_spike = volume / avg_volume if avg_volume > 0 else 0
            
            # Calculate Squeeze Score (0-100 scale)
            # 50% weight to short interest, 30% to volume spike, 20% to RSI position
            score = (min(short_float * 100, 30) * 0.5) + \
                   (min(volume_spike * 10, 30) * 0.3) + \
                   (max(0, 30 - rsi) * 0.2)  # Higher score for lower RSI
            
            scan_results.append({
                'Ticker': ticker_symbol,
                'Price': price,
                'Short % Float': f"{short_float * 100:.2f}%",
                'Volume Spike': f"{volume_spike:.1f}x",
                'Squeeze Score': f"{score:.1f}"
            })
            
        except Exception as e:
            # Skip tickers that fail but log the error
            print(f"Error processing {ticker_symbol}: {str(e)}")
            # Remove problematic tickers from the list to avoid repeated errors
            if '404' in str(e) and ticker_symbol in candidate_tickers:
                print(f"Removing {ticker_symbol} from candidate list due to 404 error")
                candidate_tickers.remove(ticker_symbol)
            continue
    
    if not scan_results:
        return pd.DataFrame()

    # Create and sort the DataFrame
    df = pd.DataFrame(scan_results)
    df_sorted = df.sort_values(by='Squeeze Score', ascending=False)
    
    # Convert score back to numeric for proper sorting in Gradio DataFrame
    df_sorted['Squeeze Score'] = pd.to_numeric(df_sorted['Squeeze Score'])
    df_sorted['Price'] = pd.to_numeric(df_sorted['Price'])
    # Convert Short % Float to numeric (removing the % sign)
    df_sorted['Short % Float'] = df_sorted['Short % Float'].str.rstrip('%').astype('float')
    # Note: There's no Short Ratio column in our dataframe
    
    return df_sorted

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="SqueezeRadarAI") as app:
    gr.Markdown("# SqueezeRadarAI: Stock Analysis & Short Squeeze Detector")
    gr.Markdown("An MVP tool to identify potential short squeeze opportunities. Data from Yahoo Finance.")

    with gr.Tab("Stock Analysis"):
        with gr.Row():
            ticker_input = gr.Textbox(label="Enter Stock Ticker", placeholder="e.g., GME")
            analyze_button = gr.Button("Analyze")
        with gr.Row():
            metrics_output = gr.Dataframe(headers=["Metric", "Value", "Description"], interactive=False)
        with gr.Row():
            news_output = gr.HTML()
        with gr.Row():
            stock_chart_output = gr.Plot()

    with gr.Tab("Short Squeeze Scanner"):
        with gr.Column():
            scanner_button = gr.Button("▶️ Run Scanner for Top Candidates")
            scanner_output = gr.DataFrame(
                headers=['Ticker', 'Price', 'Short % Float', 'Volume Spike', 'Squeeze Score'],
                datatype=['str', 'number', 'str', 'str', 'number'],
                interactive=True
            )

    # --- Event Handlers ---
    analyze_button.click(
        fn=get_stock_analysis,
        inputs=ticker_input,
        outputs=[metrics_output, news_output, stock_chart_output],
        show_progress='full'
    )

    scanner_button.click(
        fn=run_squeeze_scanner,
        inputs=None,
        outputs=scanner_output
    )

if __name__ == "__main__":
    app.launch()
