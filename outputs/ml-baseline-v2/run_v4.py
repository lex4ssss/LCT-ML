from pathlib import Path
import sys
import numpy as np
import run_v3 as v3

WEATHER_FEATURES = v3.WEATHER_FEATURES + ['precipitation_24h', 'precipitation_72h']
HOURLY = 1
MMHG_PER_HPA = 0.750062
COLUMNS = dict(temperature='temperature_2m', pressure='surface_pressure', wind_speed='wind_speed_10m', precipitation='precipitation')


def read_weather(connection, path):
    raw = connection.execute('SELECT time, ' + ', '.join(COLUMNS.values()) + ' FROM read_csv_auto(?) ORDER BY time', [str(path)]).fetchnumpy()
    weather = {name: np.ma.filled(np.ma.asarray(raw[column]).astype(np.float64), np.nan) for name, column in COLUMNS.items()}
    weather['pressure'] = weather['pressure'] * MMHG_PER_HPA
    weather['datetime'] = np.asarray(raw['time']).astype('datetime64[us]')
    return weather


def features_at(weather, moment):
    cutoff = moment - v3.AVAILABILITY_LAG
    sums = []
    for hours in (24, 72):
        span = v3.window(weather['datetime'], cutoff, hours, HOURLY)
        sums.append(np.nan if span is None else weather['precipitation'][span].sum())
    return v3.features_at(weather, moment, HOURLY) + sums


if __name__ == '__main__':
    v3.main(*map(Path, sys.argv[1:6]), reader=read_weather, features=features_at, names=WEATHER_FEATURES,
            flags=dict(precipitation=True, weather_source='open-meteo-era5-hourly', weather_lag_hours=3), candidate='hgb_weather_openmeteo')
