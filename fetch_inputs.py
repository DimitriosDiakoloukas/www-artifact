"""Fetch original third-party inputs and verify their pinned SHA-256 digests."""
from pathlib import Path
import argparse
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parent
DATA = {
    'bitcoin_alpha.csv.gz': ('https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz', '3a178611b9c2f39c9a0dc75936f28557317f1733319b49da4838232b3757cd76'),
    'bitcoin_otc.csv.gz': ('https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz', '6424ac981dad3a019f697fc1b9fcd85c19d8d9f039797758e9ffb6fea100c373'),
    'wiki_elec.txt.gz': ('https://snap.stanford.edu/data/wikiElec.ElecBs3.txt.gz', 'b62b9a64e8605b81632a66295cd4e01c99784ae46ea428c59d18deabca042c89'),
    'epinions.txt.gz': ('https://snap.stanford.edu/data/soc-sign-epinions.txt.gz', '214513a32f1375695ceab7d7581e463ebe44daa573492de699b0c5d1cf3dde60'),
    'slashdot.txt.gz': ('https://snap.stanford.edu/data/soc-sign-Slashdot090221.txt.gz', '55e69b402f2d40bc40e36c7c1211557a3d61927d5a45817f8d3db2d367080ca7'),
}
RIDGE_COMMIT = '8e89203a35fbb0462a8c1b898bb6f4747e191a39'
RIDGE_SHA256 = '6ae34e171e5518ba726c0debff5cf9dc5af6308731c1b692b47a80e8cc885b1a'


def fetch(url, path, wanted):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.download')
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            assert hashlib.sha256(temp.read_bytes()).hexdigest() == wanted, ('download hash', path.name)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == wanted, ('input hash', path.name)
    print('INPUT VERIFIED', path.relative_to(ROOT), flush=True)


def fetch_data():
    for name, (url, wanted) in DATA.items():
        fetch(url, ROOT / 'store/raw' / name, wanted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ridge', action='store_true', help='also fetch upstream RIDGE source for neural-baseline refitting')
    args = parser.parse_args()
    fetch_data()
    if args.ridge:
        fetch(f'https://raw.githubusercontent.com/Junranus/RIDGE/{RIDGE_COMMIT}/model/SGCN_ridge.py',
              ROOT / 'external/ridge/model/SGCN_ridge.py', RIDGE_SHA256)
