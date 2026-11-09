# Seconds of green light each vehicle type needs to pass the signal
VEHICLE_TIMES = {
    "car": 3,
    "motorcycle": 2,
    "bus": 5,
    "truck": 5,
}

# 2-wheelers: only timed when they are in the front row, because the ones behind
# squeeze through gaps and leave alongside the bigger vehicles.
SMALL_VEHICLES = {"motorcycle"}

# Two boxes are in the same path if they overlap horizontally by at least this
# fraction of the narrower box's width (stops slight overlaps from the next lane counting).
MIN_HORIZONTAL_OVERLAP = 0.3

# A box only counts as "in front" if its bottom edge is lower by more than this
# fraction of the other box's height (so vehicles side by side in one row don't block each other).
ROW_TOLERANCE = 0.25


def is_in_front(front, back):
    """True if box `front` is directly ahead of box `back` (camera faces the queue, stop line at the bottom)."""
    fx1, _, fx2, fy2 = front["box"]
    bx1, by1, bx2, by2 = back["box"]

    overlap = min(fx2, bx2) - max(fx1, bx1)
    narrower = min(fx2 - fx1, bx2 - bx1)
    if narrower <= 0 or overlap / narrower < MIN_HORIZONTAL_OVERLAP:
        return False

    return fy2 > by2 + ROW_TOLERANCE * (by2 - by1)


def mark_front_row(detections):
    """For small vehicles only: set detection["front_row"] = True if no vehicle (of any size) is in front of it.

    Big vehicles are always timed, so they are not checked (front_row stays None),
    but they still count as blockers for the small vehicles behind them.
    """
    for vehicle in detections:
        if vehicle["label"] not in SMALL_VEHICLES:
            vehicle["front_row"] = None
            continue
        vehicle["front_row"] = not any(
            is_in_front(other, vehicle) for other in detections if other is not vehicle
        )
    return detections


def calculate_timing(detections):
    """Work out green-light time.

    - Big vehicles (car, bus, truck): every one is timed, front row or behind.
    - Small vehicles (motorcycle): only those in the front row are timed.
    """
    rows = []
    for name, seconds_each in VEHICLE_TIMES.items():
        of_type = [d for d in detections if d["label"] == name]
        if name in SMALL_VEHICLES:
            counted = sum(1 for d in of_type if d["front_row"])
        else:
            counted = len(of_type)

        rows.append({
            "name": name,
            "detected": len(of_type),
            "counted": counted,
            "ignored": len(of_type) - counted,
            "seconds_each": seconds_each,
            "seconds_total": counted * seconds_each,
        })

    return {
        "rows": rows,
        "counted": sum(row["counted"] for row in rows),
        "total_seconds": sum(row["seconds_total"] for row in rows),
    }
