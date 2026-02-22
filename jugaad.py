from datetime import date, timedelta
from jugaad_data.nse import bhavcopy_fo_save

save_location = "D:/antigravity/option_data/"
start_date = date(2024, 8, 1)
end_date = date(2026, 2, 12)

current = start_date
while current <= end_date:
    # Skip weekends (Saturday=5, Sunday=6)
    if current.weekday() >= 5:
        current += timedelta(days=1)
        continue
    try:
        xx = bhavcopy_fo_save(current, save_location)
        print(f"Downloaded: {xx}")
    except Exception as e:
        print(f"Skipped {current}: {e}")
    current += timedelta(days=1)