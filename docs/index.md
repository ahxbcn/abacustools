# abacustools documentation

`abacustools` is a Python CLI toolkit for ABACUS DFT workflows. The documentation
is plain Markdown and is versioned with the source code.

## Start here

- [Getting started](getting-started.md)
- [Python API](python-api.md)
- [Version profiles](reference/version-profiles.md)

## Command reference

- [File commands](commands/file.md): `input`, `stru`, `interface`, `editstru`, `struinfo`, `kpt`, `traj`
- [Job commands](commands/job.md): `prepare`, `setpporb`, `checkinput`, `validate`, `status`, `monitor`, `monitor-many`
- [Postprocess commands](commands/postprocess.md): `result`, `band`, `cohp`, `molden`, `dos`, `chg`, `md`, `traj`, `mayer`, `bader`, `hirshfeld`, `ddec`
- [Workflow commands](commands/workflows.md): multi-step prepare/postprocess workflows
- [Structure databases](commands/databases.md): `database`/`db` and the `mp` spelling

## Command families

```text
abacustools file ...
abacustools job ...
abacustools postprocess ...    # aliases: post, pp
abacustools workflow ...
abacustools database ...       # alias: db
abacustools mp ...             # database --database mp
abacustools menu
```

Use `abacustools <family> --help` or `abacustools <family> <command> --help`
for the authoritative argparse help of the installed version.
