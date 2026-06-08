import nfl_data_py as nfl

try:
    rosters = nfl.import_seasonal_rosters([2025])
    print(f"Success! Rows: {len(rosters)}")
    print(rosters[rosters['position'].isin(['QB','RB','WR','TE'])].head(3).to_string())
except Exception as e:
    print(f"Failed: {e}")
