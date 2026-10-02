from backend.engine import predict_flood, severity, fingerprint

def test_prediction_probability_range():
    features={
        'rain_1h':18,'rain_6h':61,'rain_24h':126,'rain_72h':198,'forecast_rain_24h':96,
        'river_discharge_pct':91,'river_rise_6h_pct':22,'soil_moisture_pct':82,'elevation_m':1580,
        'slope_pct':2.2,'river_distance_km':0.8,'drainage_density':3.2,'population_density':10500,
        'vulnerable_population_pct':28}
    out=predict_flood(features)
    assert 0 <= out['probability'] <= 100
    assert out['horizon_hours'] == 24

def test_severity_increases_with_sos():
    a=severity(60,0.8,0); b=severity(60,0.8,4)
    assert b['score'] > a['score']

def test_ledger_chain_changes_when_payload_changes():
    a=fingerprint('GENESIS','A'); b=fingerprint('GENESIS','B')
    assert a != b
