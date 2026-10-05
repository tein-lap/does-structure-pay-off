---
name: issue_to_symbols
description: Turns the issue text into the 5 most likely starting symbols in this repository, for the graph tools.
---

# issue_to_symbols

Use this skill **once, at the start**, before exploring the code.

Run the script `scripts/issue_to_symbols.py` and pass it the parts of the issue that name code:
traceback lines, error messages, and any function, class, module or file names. You can also pass
the whole issue text.

It reads the repository in `/workspace` and prints up to 5 symbols (functions, classes or modules)
that the issue most likely refers to, best first, with their file and line. Traceback frames rank
highest, deepest frame first. Use these names as the starting point for `get_code_neighbors`,
`get_code_subgraph` or `search_similar_code`, or open them with `read_file`.

If it prints "No symbols from the issue were found", start with `run_command` and `grep -rn`.

The script uses only the Python standard library, needs no network, and takes a few seconds.
Its run time counts against your time budget.
