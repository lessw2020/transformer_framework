import time
from datetime import date


def get_date_time():
    """gets current date and time in short format...used for logging files"""
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    # Month abbreviation, day and year
    today = date.today()
    abbrev_date = today.strftime("%b-%d-%Y")

    res = abbrev_date + "-" + current_time
    return res
