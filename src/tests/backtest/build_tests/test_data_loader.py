from data_loader import load_filtered_pairs, load_price_data, prepare_pairs_data

prices = load_price_data("backtest/data/filled_data.pkl")
pairs = load_filtered_pairs("backtest/data/filtered_pairs.pkl")
print("Pairs loaded:", len(pairs))
print("Pairs sample:", list(pairs.items())[:10])

# quick prepare with relaxed thresholds to see retained pairs
pairs_data = prepare_pairs_data(prices, pairs, disable_prefilter=True, verbose=True)
print("Pairs after prepare_pairs_data:", len(pairs_data))
print("Kept sample:", list(pairs_data.keys())[:20])
