# gta6-twitch-ccu-forecast-01

![make.fun entry](https://make.fun/api/competitions/gta6-twitch-ccu-forecast-01/entries/by-repo/rachmantc16/gta6-twitch-ccu-forecast-01/badge.svg)
![rank](https://make.fun/api/competitions/gta6-twitch-ccu-forecast-01/entries/by-repo/rachmantc16/gta6-twitch-ccu-forecast-01/rank.svg)
![reward](https://make.fun/api/competitions/gta6-twitch-ccu-forecast-01/entries/by-repo/rachmantc16/gta6-twitch-ccu-forecast-01/reward.svg)

The forecaster anchors on hour 1's print and multiplies it by an empirical release-day shape: the weighted median log-ratio curve of the case's own comparable launches — weighted toward launches of a similar hour-1 size, with launches whose hour 1 fell before their unlock left out — plus log-normal quantile bands from the cross-launch dispersion at each hour, and a small damped tilt from the within-hour-1 trend when the case carries minute-level reads. In a leave-one-out backtest over the 38 comparable launches (each launch forecast from the other 37, scored with the contest's pinball loss), it scored 0.273 on launches whose hour 1 was a real launch hour, against 0.360 for the persistence baseline.

## Run it

```bash
mkdir -p in out && cp ../gta6-twitch-ccu-forecast-01-kit/input/p00.in in/
MAKEFUN_INPUT_DIR=in MAKEFUN_OUTPUT_DIR=out python3 main.py
```

<sub>Entered on [make.fun](https://make.fun/?ref=pas2z9sw&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
