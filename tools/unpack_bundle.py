"""Unpack an author-supplied data tar into a fresh workspace and verify hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path);parser.add_argument('destination', type=Path)
    parser.add_argument('--sha256', help='Expected archive hash supplied with the catalogue')
    args = parser.parse_args()
    if args.sha256:
        with args.archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest() == args.sha256
    root = args.destination.resolve();root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive) as tar:
        for member in tar:
            target = (root/member.name).resolve()
            assert target.is_relative_to(root) and member.isfile(), member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with tar.extractfile(member) as source:incoming = hashlib.file_digest(source,'sha256').hexdigest()
                with target.open('rb') as f:existing = hashlib.file_digest(f,'sha256').hexdigest()
                assert incoming == existing, f'Conflicting existing file: {target}'
            else:
                import shutil
                with tar.extractfile(member) as source, target.open('xb') as out:
                    shutil.copyfileobj(source, out, length=8*1024*1024)
    for manifest in root.rglob('BUNDLE_*.json'):
        data = json.loads(manifest.read_text());work = manifest.parent
        for name, expected in data['files'].items():
            with (work/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest() == expected, name
    print('Bundle paths and per-file checksums verified.')


if __name__ == '__main__':main()
