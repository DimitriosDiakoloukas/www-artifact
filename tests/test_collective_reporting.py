"""Integration check: completed records cannot conceal missing or altered raw measurements."""
from pathlib import Path
import json
import shutil
import tempfile
import collective.collect as collector


def test_saved_draw_integrity_gate():
    original_base,original_root=collector.BASE,collector.ROOT
    target=json.loads((original_base/'targets.json').read_text())[0]
    with tempfile.TemporaryDirectory(prefix='www-collective-integrity-') as td:
        root=Path(td);base=root/'collective';base.mkdir()
        for name in ('targets.json','LOCK.json'):
            shutil.copyfile(original_base/name,base/name)
        if (original_root/'SANITIZATION.json').exists():
            shutil.copyfile(original_root/'SANITIZATION.json',root/'SANITIZATION.json')
        record=root/target['record'];record.parent.mkdir(parents=True)
        shutil.copyfile(original_root/target['record'],record)
        folder='0-SGCN-bitcoin_alpha'
        shutil.copytree(original_base/'runs'/folder,base/'runs'/folder)
        victim=base/'runs'/folder/'q00-r1-exchange-s50.npz'
        raw=victim.read_bytes()
        try:
            collector.BASE,collector.ROOT=base,root
            victim.unlink()
            try:
                collector.collect_networks()
            except FileNotFoundError as error:
                assert error.filename==str(victim), 'fixture failed before intended missing-draw check'
            else:
                raise AssertionError('accepted missing raw draw')
            victim.write_bytes(bytes([raw[0]^1])+raw[1:])
            try:
                collector.collect_networks()
            except AssertionError:
                pass
            else:
                raise AssertionError('accepted modified raw draw')
        finally:
            collector.BASE,collector.ROOT=original_base,original_root

TESTS=[test_saved_draw_integrity_gate]
