"""Locks the replication protocol: run after PROTOCOL.md, the code and jobs.txt are committed, then
commit the LOCK.json it writes.

  python3 replication/lock.py
"""
import json
import time

from gate import CODE, GUARDED, HERE, LOCK, PROTOCOL, _git, sha256_file


def main():
    if "DRAFT" in PROTOCOL.read_text().splitlines()[0]:
        raise SystemExit("protocol still a draft")
    if _git("status", "--porcelain", "--", *GUARDED[:-1]).stdout.strip():
        raise SystemExit("commit the protocol, code and jobs.txt before locking")
    if LOCK.exists():
        raise SystemExit("already locked")
    lock = {"protocol_sha256": sha256_file(PROTOCOL), "commit": _git("rev-parse", "HEAD").stdout.strip(),
            "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_sha256": {c: sha256_file(HERE / c) for c in CODE if c != "jobs.txt"},
            "jobs_sha256": sha256_file(HERE / "jobs.txt")}
    LOCK.write_text(json.dumps(lock, indent=1) + "\n")
    print(json.dumps(lock, indent=1))
    print("now commit replication/LOCK.json")


if __name__ == "__main__":
    main()
