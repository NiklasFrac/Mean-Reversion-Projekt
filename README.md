# Mean-Reversion Pair Backtest

Kleines Research-Projekt fuer einen reproduzierbaren Pairs-Trading-Backtest: Daten laden, Paare vorfiltern, Walkforward-Fenster bauen, Mean-Reversion-Regeln testen und Ergebnisse als Dateien und Plots ablegen. Der Fokus liegt auf nachvollziehbarer Konfiguration statt Notebook-Glue.

## Start

```powershell
uv sync --extra backtest
uv run python -m download.runner_download --cfg runs/configs/config_download.yaml
uv run python -m backtest.run --config runs/configs/config_backtest.yaml
```

## Download

`download.runner_download` liest den Screener aus `runs/configs/config_download.yaml`, normalisiert Ticker, filtert unerwuenschte Securities und laedt Adjusted-Close-Daten via `yfinance`.

Zeitraum, Batches, Retry-Verhalten, Quality-Gates und Zielpfade stehen in `runs/configs/config_download.yaml`.

## Pair Prefilter

`backtest.pair_selection` sucht im Trainingsfenster nach Kandidaten mit passender Korrelation, Engle-Granger-Test, positiver Hedge-Ratio, Half-Life und Hurst-Wert.

Alle Schwellen und die maximale Pair-Anzahl kommen aus `runs/configs/config_backtest.yaml`.

## Walkforward

`backtest.walkforward` trennt Train- und Test-Zeitraeume anhand der Backtest-Config. Pro Fenster werden Paare neu selektiert und Strategieparameter neu festgelegt.

Fenstermodus, Laengen, Schrittweite sowie Start und Ende stehen in `runs/configs/config_backtest.yaml`.

## Mean Rev

Die Strategie handelt den Spread eines Pairs ueber einen rollierenden Z-Score. Entries, Exits, Stops, Cooldown und maximale Haltedauer werden aus der Config gelesen.

Kosten und Positionsgewichtung sind ebenfalls config-getrieben.

## BO

Wenn `bo.enabled` aktiv ist, optimiert `backtest.optimize` die Strategy-Parameter je Walkforward-Fenster auf dem Trainingsbereich. Der Suchraum und die Anzahl der Versuche stehen in `runs/configs/config_backtest.yaml`.

Falls Bayesian Optimization nicht verfuegbar ist, faellt der Code auf eine Random Search mit derselben Config zurueck.

## Output

Download-Outputs:

- `raw_close.csv`
- `filled_close.csv`
- `dropped_tickers.csv`

Die Pfade stehen in `runs/configs/config_download.yaml`.

Backtest-Outputs im konfigurierten `output.dir`:

- `summary.json`
- `config_used.yaml`
- `daily.csv`, `equity.csv`
- `positions.csv`, `weights.csv`, `trades.csv`
- `windows.csv`, `selected_pairs.csv`
- `bo_trials.csv`, `bo_best.json`
- Plots: `equity.png`, `drawdown.png`, `params.png`
