# Seconds of green light each vehicle type needs to pass the signal
VEHICLE_TIMES = {
    "car": 3,
    "motorcycle": 2,
    "bus": 5,
    "truck": 5,
}


def calculate_timing(counts):
    """Work out the time each vehicle type needs and the total green-light time."""
    rows = []
    for name, count in counts.items():
        seconds_each = VEHICLE_TIMES.get(name, 0)
        rows.append({
            "name": name,
            "count": count,
            "seconds_each": seconds_each,
            "seconds_total": count * seconds_each,
        })

    return {"rows": rows, "total_seconds": sum(row["seconds_total"] for row in rows)}
