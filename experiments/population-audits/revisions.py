"""Raw-data evidence of rating revision: ordered pairs carrying several judgments."""
from pathlib import Path
import collections, hashlib, json, sys
HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
sys.path[:0] = [str(CAMPAIGN / 'src'), str(CAMPAIGN)]
from srange.data.snap import SNAP, _triples, raw_path, RAW_SHA256, sha256_file

def run():
    result = {}
    for net in ('bitcoin_alpha', 'bitcoin_otc', 'wiki_elec', 'epinions', 'slashdot'):
        assert sha256_file(raw_path(net)) == RAW_SHA256[net]
        ratings = collections.defaultdict(list)
        for source, target, rating in _triples(raw_path(net), SNAP[net][1]):
            if source != target:
                ratings[source, target].append(rating)
        repeated = [v for v in ratings.values() if len(v) > 1]
        result[net] = {'ordered_pairs': len(ratings), 'repeated_pairs': len(repeated), 'both_signs': sum((max(v) > 0 > min(v) for v in repeated))}
    (HERE / 'REVISIONS.json').write_text(json.dumps({'networks': result, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2, sort_keys=True) + '\n')
    print(result)
if __name__ == '__main__':
    run()
