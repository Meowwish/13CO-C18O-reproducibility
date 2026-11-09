"""Check frozen edit inputs once, before the batch job modifies the paper."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT/'input_manifest.json').read_text())
    for path, digest in manifest['hashes'].items():
        with Path(path).open('rb') as f: assert hashlib.file_digest(f, 'sha256').hexdigest() == digest, path
    assert (ROOT.parent/'Paper/mnras_template.tex').read_bytes() == (ROOT/'mnras_template.before.tex').read_bytes()
    print('Paper input freeze verified.', flush=True)


if __name__ == '__main__': main()
