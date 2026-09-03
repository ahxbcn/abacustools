Collection of tools used for performing DFT calculation with ABACUS.

## Command-line subcommands

Subcommands are registered in the package with Python's
`argparse.add_subparsers()`. The top-level command currently provides:

```text
abacustools version
```

Both the top-level parser and each subcommand provide their own help text:

```text
abacustools --help
abacustools version --help
```
