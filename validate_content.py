"""Content validator CLI (spec §73).

    python validate_content.py            # prints issues, exit 1 if any error

Checks duplicate ids, missing titles, broken prerequisites / related / lab /
term links, missing previous/next, orphan lessons, glossary → lesson links,
prerequisite cycles, and lessons without a render() function.
"""

import sys

from core.registry import get_registry


def main() -> int:
    reg = get_registry()
    issues = reg.validate()
    print(f"sections={len(reg.sections)} modules={len(reg.modules)} lessons={len(reg.lessons)} "
          f"labs={len(reg.labs)} terms={len(reg.terms)}")
    for issue in issues:
        print(issue)
    errors = [i for i in issues if i.level == "error"]
    print(f"{len(errors)} error(s), {len(issues) - len(errors)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
