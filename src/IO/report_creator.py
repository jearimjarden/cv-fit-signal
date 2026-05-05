import csv
from pathlib import Path


def save_report(reports: list[dict], save_path: str, save_name: str):
    path = Path(save_path) / save_name

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reports[0].keys())
        writer.writeheader()
        writer.writerows(reports)
