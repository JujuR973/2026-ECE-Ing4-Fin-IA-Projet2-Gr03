trades = [
    ("AAPL", "BUY",  100),
    ("GOOG", "BUY",   50),
    ("AAPL", "SELL",  60),
    ("TSLA", "BUY",  200),
    ("GOOG", "BUY",   30),
    ("AAPL", "BUY",   40),
]

# Attendu : {"AAPL": 80, "GOOG": 80, "TSLA": 200}
positions = {}
for asset,side,qty in trades :
    signe = 1 if side == "BUY" else -1
    if asset in positions: 
        positions[asset] += signe*qty
    else : 
        positions[asset]=signe*qty


print (positions) 