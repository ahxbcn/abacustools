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

Calculation results can be collected from one or more completed ABACUS jobs:

```text
abacustools postprocess result -j JOB -p energy force stress
```

Use `abacustools postprocess result --help` to see all supported result
parameters. If `--param` is omitted, the command selects the results relevant
to the job's calculation type and displays scalar results as compact,
borderless tables. If the result columns do not fit the terminal, they are
split into multiple tables with repeated headers rather than being truncated
or wrapped by the terminal. A result that does not apply to a job is shown as
`-`. Large array results such as `force` and `stress` are omitted from this
summary.
