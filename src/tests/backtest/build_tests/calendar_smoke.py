# scripts/test_calendar.py
import datetime as dt
import sys


def main():
    try:
        import exchange_calendars as xcals
    except Exception as e:
        print("FEHLER: exchange_calendars nicht importierbar ->", e)
        sys.exit(1)

    cal = xcals.get_calendar("XNYS")
    start, end = "2024-12-15", "2025-01-15"
    sessions = cal.sessions_in_range(start, end)
    assert len(sessions) > 0, "Keine Handelstage im Intervall gefunden"

    # sanity checks
    christmas = dt.date(2024, 12, 25)
    assert not cal.is_session(christmas), "Christmas sollte kein Handelstag sein"
    print(
        "OK:",
        cal.name,
        "| Tage:",
        len(sessions),
        "| first:",
        sessions[0].date(),
        "| last:",
        sessions[-1].date(),
    )
    print("next_open after Christmas:", cal.next_open(christmas))
    print("previous_close before Christmas:", cal.previous_close(christmas))


if __name__ == "__main__":
    main()
