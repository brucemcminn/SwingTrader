for sym in tqdm(symbols):
    try:
        df = fetch_ohlc(sym, interval="1d", lookback_days=365)
    except:
        print("fail:", sym)
