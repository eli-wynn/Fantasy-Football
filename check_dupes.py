import nfl_data_py as nfl

picks = nfl.import_draft_picks()
print("Columns:", picks.columns.tolist())
