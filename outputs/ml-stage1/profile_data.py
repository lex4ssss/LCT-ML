import argparse
import json
import sys
from pathlib import Path

from audit import profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--channels', type=Path, required=True)
    parser.add_argument('--facilities', type=Path, required=True)
    parser.add_argument('--source-timezone', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        columns = json.loads(Path(__file__).with_name('columns.json').read_text(encoding='utf-8'))
        report = profile(args.events, args.channels, args.facilities, args.source_timezone, columns)
        report['run'] = dict(streaming=True, reject_invalid_rows=True, remove_duplicates=False,
                             numeric_classification=False, replay=False, synthetic_data=False,
                             training=False, inference=False, schema_version=1)
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            with args.output.open('x', encoding='utf-8') as handle:
                handle.write(payload)
        else:
            sys.stdout.write(payload)
        return 2 if report['rejected_rows'] or report['unmapped_rows'] else 0
    except (OSError, ValueError) as exc:
        print('Audit failed: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
