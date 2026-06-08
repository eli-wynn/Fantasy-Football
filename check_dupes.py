import nfl_data_py as nfl

sched = nfl.import_schedules([2024])

# Look at a few rows with the relevant columns
cols = ['season', 'week', 'away_team', 'home_team', 'spread_line', 'total_line', 'temp', 'wind']
sample = sched[cols].dropna(subset=['spread_line', 'total_line']).head(5)
print(sample.to_string())
print(f"\nTotal games: {len(sched)}")
print(f"Games with lines: {sched['total_line'].notna().sum()}")
