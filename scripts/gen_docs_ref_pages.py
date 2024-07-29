"""Generate the code reference pages."""

import fnmatch
import os
from pathlib import Path

import mkdocs_gen_files

root = Path(__file__).parent.parent
src = root / "futureframe"
prefix = "futureframe"
# remove docs/reference directory
if os.path.exists(root / "docs/reference"):
    os.system("rm -r " + str(root / "docs/reference/*"))


print(f"Generating reference pages from {src}")

ignore_list = [
    "_*",
    "__init__",
    "__main__",
    "_registry",
    "types",
]

for path in sorted(src.rglob("*.py")):
    print(f"Generating reference page for {path}")
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")

    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)
    parts = (prefix, *parts)

    if parts[-1] in ignore_list:
        continue

    if any(fnmatch.fnmatch(parts[-1], pattern) for pattern in ignore_list):
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))