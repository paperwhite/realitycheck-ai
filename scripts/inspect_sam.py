from pathlib import Path
from meta_sam_parser import parse_responses_stream

raw = Path("sam_response.txt").read_text()

try:
    result = parse_responses_stream(raw)
except Exception as exc:
    print("Parser error:", type(exc).__name__, str(exc))
    raise

print("Parsed result type:", type(result).__name__)

if isinstance(result, dict):
    print("Top-level keys:", list(result.keys()))

elif isinstance(result, (list, tuple)):
    print("Number of entries:", len(result))

    if result:
        first = result[0]
        print("First entry type:", type(first).__name__)

        if isinstance(first, dict):
            print("First entry keys:", list(first.keys()))
        else:
            print("First entry attributes:",
                  [x for x in dir(first) if not x.startswith("_")])
else:
    print("Result attributes:",
          [x for x in dir(result) if not x.startswith("_")])


