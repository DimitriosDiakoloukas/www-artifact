"""Class-balanced metrics for the frozen gated/population operating points."""
from pathlib import Path
import csv
import json
from balanced import Intervals, rep, pr
HERE = Path(__file__).resolve().parent

def main():
    result = {}
    flat = []
    for key, folder, prefix in [(f'random{u}', pr.V6 / f'runs/v1/unit{u}', 'shard') for u in range(6)] + [(net, pr.V6 / f'runs/scale-v1/{net}', 'audit') for net in ('epinions', 'slashdot')]:
        rows = rep.v6_rows(folder, prefix)
        assert rows and (not any((r.get('unresolved') for r in rows)))
        result[key] = {}
        for head in rows[0]['heads']:
            values = Intervals(rows, head).rates()
            result[key][head] = values
            flat.append(dict(unit=key, model=head, **values))
        print('GATED BALANCED', key, len(rows), flush=True)
    (HERE / 'BALANCED_GATED.json').write_text(json.dumps(result, indent=2) + '\n')
    with (HERE / 'BALANCED_GATED.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
if __name__ == '__main__':
    main()
