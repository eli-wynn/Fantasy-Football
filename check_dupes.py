import nfl_data_py as nfl

rosters = nfl.import_seasonal_rosters([2024])
print("Columns:", rosters.columns.tolist())
print("\nSample row:")
print(rosters.iloc[0].to_dict())
