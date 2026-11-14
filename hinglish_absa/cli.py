"""
Command Line Interface (CLI) for Hinglish Aspect-Based Sentiment Analysis.
"""

import sys
import json
import argparse
from hinglish_absa import HinglishABSA

def main():
    parser = argparse.ArgumentParser(
        description="Hinglish Aspect-Based Sentiment Analysis CLI"
    )
    parser.add_argument(
        "text",
        type=str,
        nargs="?",
        help="Input Hinglish sentence to analyze"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to text file containing one Hinglish sentence per line"
    )

    args = parser.parse_args()

    absa = HinglishABSA()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        results = absa.analyze_batch(lines)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
        else:
            for r in results:
                print(r.summary())
                print("-" * 50)
        return

    if not args.text:
        print("Please provide a Hinglish sentence or use --help. Example:")
        print('  python -m hinglish_absa.cli "Phone ka camera mast hai but battery bekar hai."')
        sys.exit(1)

    result = absa.analyze(args.text)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.summary())

if __name__ == "__main__":
    main()
