from backend.db import engine
from sqlalchemy import text
import nfl_data_py as nfl

# Check what nfl_data_py weekly data has for headshots
df = nfl.import_weekly_data([2024])
sample = df[df['player_display_name'].str.contains('Mahomes', na=False)][['player_id', 'player_display_name', 'headshot_url']].head(3)
print(sample)
