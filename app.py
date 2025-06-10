import os
import gradio as gr
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import sys

def load_environment():
    """Load environment variables and verify API key."""
    # Print current working directory for debugging
    print("\n=== Environment Setup ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Try to load .env file from the same directory as app.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    print(f"Looking for .env file at: {env_path}")
    
    # Check if .env file exists
    if not os.path.exists(env_path):
        print("ERROR: .env file not found!")
        print(f"Please create a .env file at: {env_path}")
        print("The file should contain: FMP_API_KEY=your_api_key_here")
        return None
    
    # Load environment variables from .env file
    load_dotenv(env_path)
    
    # Get FMP API key from environment variables
    api_key = os.getenv('FMP_API_KEY')
    
    # Debug output
    print("\n=== Environment Variables ===")
    print(f"FMP_API_KEY found: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"API Key length: {len(api_key)} characters")
        print(f"Key starts with: {api_key[:4]}...")
    else:
        print("ERROR: FMP_API_KEY not found in .env file")
        print("Please make sure your .env file contains: FMP_API_KEY=your_key_here")
    
    return api_key

# Load API key
FMP_API_KEY = load_environment()

# If we don't have an API key, we'll use a fallback mode
if not FMP_API_KEY:
    print("\nWARNING: Running in fallback mode with limited functionality")
    print("Please set up your FMP API key in the .env file for full features.")

def get_stock_analysis(ticker):
    if not ticker:
        return pd.DataFrame(), "<span style='color:red;'>Please enter a stock ticker.</span>", None

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # --- Data Fetching and Validation ---
        if info.get('regularMarketPrice') is None:
             return pd.DataFrame(), f"<span style='color:red;'>Could not find data for ticker '{ticker}'. Please check the symbol.</span>", None

        # Fetch historical data for the last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return pd.DataFrame(), f"<span style='color:red;'>No historical data found for {ticker}.</span>", None

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

        return metrics_df, news_html, fig

    except Exception as e:
        return pd.DataFrame(), f"<span style='color:red;'>An error occurred: {e}</span>", None

def fetch_short_interest_data():
    """Fetch short interest data from FMP API."""
    if not FMP_API_KEY:
        print("Warning: FMP_API_KEY not found in environment variables")
        return pd.DataFrame()
    
    try:
        url = f"https://financialmodelingprep.com/api/v3/stock/short-interest?apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        print(f"Error fetching short interest data: {e}")
        return pd.DataFrame()

def run_squeeze_scanner():
    """Run the short squeeze scanner using FMP API and yfinance data."""
    print("\n=== Starting Scanner ===")
    results = []
    
    # Expanded list of potential squeeze candidates
    potential_tickers = [
        # Meme stocks
        'GME', 'AMC', 'BB', 'BBBYQ', 'KOSS', 'NOK', 'CLOV', 'WISH', 'RIDE',
        # High short interest stocks
        'BYND', 'CVNA', 'UPST', 'AFRM', 'RIVN', 'LCID', 'MULN', 'RDBX', 'REV',
        # Popular tech
        'TSLA', 'NVDA', 'AMD', 'PLTR', 'SOFI', 'HOOD', 'COIN', 'MSTR', 'MARA', 'RIOT'
    ]
    
    print(f"Scanning {len(potential_tickers)} potential tickers...")
    
    try:
        # Try to get FMP data but don't fail if it doesn't work
        try:
            short_interest_df = fetch_short_interest_data()
            if not short_interest_df.empty:
                print(f"Found {len(short_interest_df)} stocks with short interest data")
                # Add top FMP tickers to our list
                fmp_tickers = short_interest_df.nlargest(15, 'shortPercent')['symbol'].tolist()
                potential_tickers = list(set(potential_tickers + fmp_tickers))
                print(f"Added {len(fmp_tickers)} tickers from FMP")
        except Exception as e:
            print(f"Couldn't fetch FMP data, using default list: {e}")
        
        # Process each ticker
        for ticker in potential_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Skip if no price data or price too low
                if not info or 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
                    continue
                    
                price = info.get('regularMarketPrice', 0)
                if price < 1:  # Skip penny stocks
                    continue
                
                # Calculate volume spike (today's volume vs 20-day average)
                hist = stock.history(period="21d")  # 21 days to calculate 20-day average
                if len(hist) < 5:  # Skip if not enough data
                    continue
                    
                avg_volume = hist['Volume'][:-1].mean()  # Exclude today
                volume_today = hist['Volume'][-1] if not hist.empty else 0
                volume_spike = volume_today / avg_volume if avg_volume > 0 else 1
                
                # Get short interest data from yfinance
                short_float = info.get('shortPercentOfFloat', 0) * 100
                short_ratio = info.get('shortRatio', 0)
                
                # If we have FMP data, use it
                if 'short_interest_df' in locals() and not short_interest_df.empty:
                    fmp_data = short_interest_df[short_interest_df['symbol'] == ticker]
                    if not fmp_data.empty and 'shortPercent' in fmp_data.columns:
                        short_float = float(fmp_data['shortPercent'].iloc[0])
                
                # Calculate squeeze score (0-100)
                # Higher short float and volume spike = higher score
                squeeze_score = min(100, (float(short_float) * 2) + (min(volume_spike, 5) * 10))
                
                # Only include stocks with significant short interest
                if float(short_float) < 5:  # At least 5% short interest
                    continue
                
                results.append({
                    'Ticker': ticker,
                    'Price': f"${price:.2f}",
                    'Short % Float': f"{float(short_float):.2f}%",
                    'Volume Spike': f"{volume_spike:.1f}x",
                    'Squeeze Score': round(squeeze_score, 1)
                })
                
                print(f"Added {ticker}: {short_float:.1f}% short, {volume_spike:.1f}x volume")
                
                # Limit to top 15 results for performance
                if len(results) >= 15:
                    break
                    
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in scanner: {e}")
        # If we have some results, return them even if there was an error
        if not results:
            return pd.DataFrame(columns=['Ticker', 'Price', 'Short % Float', 'Volume Spike', 'Squeeze Score'])
    
    # Sort by squeeze score (highest first)
    if not results:
        print("No stocks met the criteria")
        return pd.DataFrame(columns=['Ticker', 'Price', 'Short % Float', 'Volume Spike', 'Squeeze Score'])
    
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('Squeeze Score', ascending=False)
    
    print(f"\n=== Found {len(df_sorted)} potential squeeze candidates ===")
    print(df_sorted[['Ticker', 'Squeeze Score', 'Short % Float', 'Volume Spike']].to_string())
    
    return df_sorted
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
