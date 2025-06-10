---
title: SqueezeRadarAI
emoji: 🚀
colorFrom: pink
colorTo: blue
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
pinned: false
---

# SqueezeRadarAI 🚀

Welcome to SqueezeRadarAI, a simple yet powerful tool for identifying potential short squeeze opportunities in the stock market. This application is built with Python, Gradio, and `yfinance`, and is hosted on Hugging Face Spaces.

## Features

-   **🔍 Single Stock Analysis:** Enter any stock ticker to get key metrics like current price, RSI, short interest, and trading volume. View an interactive price chart with Bollinger Bands.
-   **📡 Short Squeeze Scanner:** Run a scan on a curated list of highly-shorted stocks. The scanner provides a "Squeeze Score" to rank potential candidates, based on a combination of high short interest and non-overbought technicals.
-   **📊 Interactive Data:** The scanner results are displayed in a sortable table, allowing you to quickly identify the top candidates.

## How to Use

1.  **Stock Analysis Tab:**
    -   Type a valid stock ticker (e.g., `GME`, `TSLA`) into the input box.
    -   Click "Analyze".
    -   The key metrics and an interactive chart will appear.

2.  **Short Squeeze Scanner Tab:**
    -   Click the "▶️ Run Scanner for Top Candidates" button.
    -   Please wait as the app fetches data for multiple stocks. A progress bar will be displayed.
    -   The table will populate with results, sorted by the highest Squeeze Score.

## Disclaimer

This tool is for educational and informational purposes only. It is not financial advice. All data is provided by Yahoo Finance and may have delays or inaccuracies. Always do your own research before making any investment decisions.
