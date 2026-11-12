"""Download Samsung INR21700-30T dataset (.mat files + docs) from Mendeley Data 9xyvy2njj3 v2.

Source: Kollmeyer et al., McMaster University, CC BY 4.0.
CSV duplicates of the .mat files are skipped.
"""
import json
import pathlib
import urllib.request

DATASET = "9xyvy2njj3"
VERSION = 2
API = "https://data.mendeley.com/public-api/datasets"
UA = {"User-Agent": "Mozilla/5.0 (dataset-downloader)"}
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "samsung_30t"


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r:
        return json.load(r)


def fetch(f, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == f["size"]:
        return
    req = urllib.request.Request(f["content_details"]["download_url"], headers=UA)
    with urllib.request.urlopen(req) as r:
        dest.write_bytes(r.read())


def main():
    for f in get_json(f"{API}/{DATASET}/files?folder_id=root&version={VERSION}"):
        if f["filename"].endswith((".txt", ".pdf")) or "30T" in f["filename"]:
            fetch(f, OUT / f["filename"])
    for folder in get_json(f"{API}/{DATASET}/folders/{VERSION}"):
        files = get_json(f"{API}/{DATASET}/files?folder_id={folder['id']}&version={VERSION}")
        for f in files:
            if f["filename"].endswith(".mat"):
                fetch(f, OUT / folder["name"] / f["filename"])
    print("done:", OUT)


if __name__ == "__main__":
    main()
