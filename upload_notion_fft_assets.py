"""Send the exact multipart requests returned by Notion's file upload tool."""
import concurrent.futures
import json
from pathlib import Path
import sys

import requests


def upload(item):
    metadata = item["upload"]
    with Path(item["path"]).open("rb") as stream:
        response = requests.post(metadata["upload_url"], headers=metadata["upload_headers"],
                                 files={"file": (metadata["filename"], stream, "image/png")},
                                 timeout=60)
    response.raise_for_status()
    data = response.json()
    return dict(image=item["image"], kind=item["kind"], result=data)


def main():
    config = Path(sys.argv[1])
    output = Path(sys.argv[2])
    manifest = json.loads(config.read_text(encoding="utf-8"))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        pending = {executor.submit(upload, item): item for item in manifest}
        for future in concurrent.futures.as_completed(pending):
            item = pending[future]
            try:
                results.append(future.result())
                print(f"Uploaded {item['image']} / {item['kind']}", flush=True)
            except Exception as exc:
                # Do not print bearer headers or signed upload URLs.
                results.append(dict(image=item["image"], kind=item["kind"], error=type(exc).__name__))
                print(f"Upload failed {item['image']} / {item['kind']}: {type(exc).__name__}", flush=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if any("error" in row for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
