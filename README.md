Collection of tools used for performing DFT calculation with ABACUS.

## Command-line subcommands

Subcommands are registered in the package with Python's
`argparse.add_subparsers()`. The top-level command provides `version` and
nested `file` subcommands:

```text
abacustools version
abacustools file input INPUT
abacustools file stru STRU
abacustools file kpt KPT
```

Both the top-level parser and each subcommand provide their own help text:

```text
abacustools --help
abacustools file --help
abacustools file input --help
```
