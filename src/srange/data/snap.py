"""SNAP signed networks: download, verify, parse.

Parsers are vendored from the old harness (`load_signed`, see PROVENANCE.md); PROCESSED_SHA256 pins
the exact output of that function, so any change in parsing fails loudly instead of drifting.
Preprocessing: an undirected relation per unordered pair, signed by the sum of its ratings; pairs
summing to zero and self-loops are dropped; node ids are remapped in sorted order of original id.
"""
from __future__ import annotations

import gzip
import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from srange.paths import STORE

PREPROCESSING_VERSION = "undirected-summed-sign-v1"

SNAP = {   # name -> (url, format)
    "bitcoin_alpha": ("https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz", "bitcoin"),
    "bitcoin_otc": ("https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz", "bitcoin"),
    "wiki_rfa": ("https://snap.stanford.edu/data/wiki-RfA.txt.gz", "wikirfa"),
    "wiki_elec": ("https://snap.stanford.edu/data/wikiElec.ElecBs3.txt.gz", "wikielec"),
    "slashdot": ("https://snap.stanford.edu/data/soc-sign-Slashdot090221.txt.gz", "edgelist"),
    "epinions": ("https://snap.stanford.edu/data/soc-sign-epinions.txt.gz", "edgelist"),
}

# Raw files as downloaded from SNAP in August 2026.
RAW_SHA256 = {
    "bitcoin_alpha": "3a178611b9c2f39c9a0dc75936f28557317f1733319b49da4838232b3757cd76",
    "bitcoin_otc": "6424ac981dad3a019f697fc1b9fcd85c19d8d9f039797758e9ffb6fea100c373",
    "wiki_rfa": "88d53196fb2564a2e20286dbba818832f718cc352bb181a2101d23d2556f0862",
    "wiki_elec": "b62b9a64e8605b81632a66295cd4e01c99784ae46ea428c59d18deabca042c89",
    "slashdot": "55e69b402f2d40bc40e36c7c1211557a3d61927d5a45817f8d3db2d367080ca7",
    "epinions": "214513a32f1375695ceab7d7581e463ebe44daa573492de699b0c5d1cf3dde60",
}

# Output of the old `load_signed` on the raw files above (hash of n, edges int64, signs float32).
PROCESSED_SHA256 = {
    "bitcoin_alpha": "c6b2648b58f956f6f07128f9110d5018af147d617fc77ef05a3b0869686603ff",
    "bitcoin_otc": "ac7cb945a28dcc4dbe5f35dc8a136e231555f38fec68274b0e0bbba0afd7b36a",
    "wiki_rfa": "5b1e915cd252f6f7d537c1e8dea752f9256846e0c25b8e2b20700df3417fb1dc",
    "wiki_elec": "dd525f30ac3857123ce0c73b6fb177f63776fac9decfd79fb3a8728fb2e0573b",
    "slashdot": "2f4a50e0f992479e33dda6051fa04d18027a8b910b9f2deb3d809c0319eef6f2",
    "epinions": "bc8e3ccef5cc23626aa0cac61e42dd5c73f36d3b3a21d735fae9f4348b3d7b48",
}


@dataclass
class SignedEdges:
    """Undirected signed network: edges [E, 2] with edges[:, 0] < edges[:, 1], signs [E] in {-1, +1}."""
    name: str
    n: int
    edges: np.ndarray
    signs: np.ndarray
    meta: dict = field(default_factory=dict)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def processed_sha256(n, edges, signs) -> str:
    h = hashlib.sha256()
    h.update(str(int(n)).encode())
    h.update(np.ascontiguousarray(edges.astype(np.int64)).tobytes())
    h.update(np.ascontiguousarray(signs.astype(np.float32)).tobytes())
    return h.hexdigest()


def raw_path(name: str) -> Path:
    url, fmt = SNAP[name]
    return STORE / "raw" / (name + (".csv.gz" if fmt == "bitcoin" else ".txt.gz"))


def _triples(path: Path, fmt: str):
    if fmt == "bitcoin":                                   # src,tgt,rating,time
        raw = np.loadtxt(gzip.open(path, "rt"), delimiter=",")
        return [(int(s), int(t), float(r)) for s, t, r, _ in raw]
    if fmt == "edgelist":                                  # from to sign, '#' comments
        raw = np.loadtxt(gzip.open(path, "rt"), comments="#", dtype=np.int64)
        return [(int(s), int(t), float(r)) for s, t, r in raw]
    if fmt == "wikirfa":                                   # SRC:/TGT:/VOT: blocks
        triples, s, t = [], None, None
        for line in gzip.open(path, "rt", encoding="utf-8", errors="ignore"):
            if line.startswith("SRC:"):
                s = line[4:].strip()
            elif line.startswith("TGT:"):
                t = line[4:].strip()
            elif line.startswith("VOT:"):
                try:
                    r = int(line[4:].strip())
                except ValueError:
                    r = 0
                if s and t and r != 0:
                    triples.append((s, t, float(r)))
                s = t = None
        return triples
    if fmt == "wikielec":                                  # U = candidate, V = vote
        triples, cand = [], None
        for line in gzip.open(path, "rt", encoding="latin-1", errors="ignore"):
            f = line.rstrip("\n").split("\t")
            if f[0] == "U" and len(f) >= 2:
                cand = f[1].strip()
            elif f[0] == "V" and len(f) >= 3 and cand is not None:
                try:
                    r, voter = int(f[1]), f[2].strip()
                except ValueError:
                    continue
                if r != 0 and voter != cand:
                    triples.append((voter, cand, float(r)))
        return triples
    raise ValueError(f"unknown format {fmt!r}")


def load_snap(name: str, download: bool = False) -> SignedEdges:
    url, fmt = SNAP[name]
    path = raw_path(name)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} missing; rerun with download=True")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    got = sha256_file(path)
    if got != RAW_SHA256[name]:
        raise ValueError(f"{name}: raw sha256 {got} != pinned {RAW_SHA256[name]}")
    pair = {}
    for s, t, r in _triples(path, fmt):
        if s == t:
            continue
        k = (min(s, t), max(s, t))
        pair[k] = pair.get(k, 0.0) + r
    ids = sorted({v for k in pair for v in k})
    remap = {v: i for i, v in enumerate(ids)}
    edges, signs = [], []
    for (a, b), r in pair.items():
        if r == 0:
            continue
        edges.append((remap[a], remap[b]))
        signs.append(1.0 if r > 0 else -1.0)
    n = len(ids)
    edges = np.array(edges, dtype=np.int64)
    signs = np.array(signs, dtype=np.float32)
    ph = processed_sha256(n, edges, signs)
    if ph != PROCESSED_SHA256[name]:
        raise ValueError(f"{name}: processed sha256 {ph} != pinned {PROCESSED_SHA256[name]}")
    # min(s,t) on original ids and a sorted remap keep edges[:, 0] < edges[:, 1]
    assert (edges[:, 0] < edges[:, 1]).all()
    return SignedEdges(name, n, edges, signs, meta={
        "url": url, "raw_sha256": got, "processed_sha256": ph,
        "preprocessing": PREPROCESSING_VERSION})
