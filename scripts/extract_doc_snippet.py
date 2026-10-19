#!/usr/bin/env python3
"""Print the body of the one fenced code block in a markdown file that carries a
given fence language and contains a given literal substring.

The selection must be unambiguous. A caller executing documented text needs to
know it got the block the document means, so no match and several matches are
both hard failures with distinct exit codes, and a block whose body is blank
counts as no match at all.

Usage:
    extract_doc_snippet.py --file <markdown> --lang <fence language> --contains <substring>

Exit codes:
    0  exactly one block matched; its body is on stdout, fence lines excluded
    1  the file could not be read
    2  no non-blank block of that language contains the substring
    3  more than one block of that language contains the substring
"""

import sys
import argparse

FENCE = "```"


def fenced_blocks(lines):
    lang, body = None, []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith(FENCE):
            if lang is not None:
                body.append(line)
            continue
        if lang is None:
            info = stripped[len(FENCE):].split()
            lang, body = (info[0] if info else ""), []
        elif stripped == FENCE:
            yield lang, "".join(body)
            lang = None
        else:
            body.append(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--file", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--contains", required=True)
    args = parser.parse_args()

    try:
        with open(args.file, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as err:
        print(f"extract_doc_snippet: {err}", file=sys.stderr)
        return 1

    matches = [body for lang, body in fenced_blocks(lines)
               if lang == args.lang and body.strip() and args.contains in body]
    where = f"{args.file}: ```{args.lang} block containing {args.contains!r}"
    if not matches:
        print(f"extract_doc_snippet: no {where}", file=sys.stderr)
        return 2
    if len(matches) > 1:
        print(f"extract_doc_snippet: {len(matches)} matches for {where}; expected exactly one",
              file=sys.stderr)
        return 3
    sys.stdout.write(matches[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
