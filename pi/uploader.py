"""Push detection records from the Pi to a hosted database.

The Pi makes an outbound connection and nothing reaches in, so no port has to
be opened on the router and the Pi never becomes internet facing.

    export SUPABASE_URL="https://xxxxxxxx.supabase.co"
    export SUPABASE_KEY="<anon or service key>"

    python uploader.py            # send anything new, then exit
    python uploader.py --watch    # keep sending every 30 s

Credentials are read from the environment. Do not put them in this file, and
do not commit a .env that contains them.

Table to create in Supabase first:

    create table detections (
      id          bigserial primary key,
      device_id   text        not null,
      ts          timestamptz not null,
      cls         text        not null,
      conf        real        not null,
      x1 int, y1 int, x2 int, y2 int,
      belt_speed  real,
      event       text,
      created_at  timestamptz default now()
    );
    create index on detections (device_id, ts desc);
"""

import argparse
import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(HERE, "records.jsonl")
STATE = os.path.join(HERE, ".upload_state")

BATCH = 200
TABLE = os.environ.get("SUPABASE_TABLE", "detections")


def last_sent():
    try:
        with open(STATE) as fh:
            return int(fh.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def save_sent(n):
    with open(STATE, "w") as fh:
        fh.write(str(n))


def unsent_rows(after_id):
    """Read records newer than the last uploaded id.

    The id is the watermark rather than a timestamp, because ids increase
    strictly while two records can share a second.
    """
    if not os.path.exists(RECORDS):
        return []
    rows = []
    with open(RECORDS, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue                      # skip a torn final line
            if r.get("id", 0) > after_id:
                rows.append(r)
    return rows


def to_payload(r):
    box = r.get("box") or [0, 0, 0, 0]
    return {
        "device_id": r.get("device", "unknown"),
        "ts": r.get("ts"),
        "cls": r.get("cls"),
        "conf": r.get("conf"),
        "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
        "belt_speed": r.get("speed"),
        "event": r.get("event") or None,
    }


def push(url, key, rows):
    endpoint = "%s/rest/v1/%s" % (url.rstrip("/"), TABLE)
    headers = {
        "apikey": key,
        "Authorization": "Bearer %s" % key,
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    resp = requests.post(endpoint, headers=headers,
                         data=json.dumps([to_payload(r) for r in rows]),
                         timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError("%s %s" % (resp.status_code, resp.text[:300]))


def run_once(url, key, verbose=True):
    after = last_sent()
    rows = unsent_rows(after)
    if not rows:
        if verbose:
            print("nothing new (watermark id=%d)" % after)
        return 0

    sent = 0
    # Send in batches and move the watermark after each one, so an interruption
    # costs at most one batch of duplicates rather than restarting from zero.
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        push(url, key, chunk)
        sent += len(chunk)
        save_sent(chunk[-1]["id"])
        if verbose:
            print("sent %d rows, watermark id=%d"
                  % (len(chunk), chunk[-1]["id"]))
    return sent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=int, default=30)
    a = p.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        sys.exit("Set SUPABASE_URL and SUPABASE_KEY in the environment.")

    if not a.watch:
        run_once(url, key)
        return

    print("watching %s every %ds" % (RECORDS, a.interval))
    while True:
        try:
            run_once(url, key, verbose=False)
        except Exception as exc:
            # Never die on a network blip. The watermark has not moved, so the
            # same rows are retried next time round.
            print("upload failed, will retry: %s" % exc, file=sys.stderr)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
