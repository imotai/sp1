# cuSlop

A cuda version of slop.

## Cargo profiles

To use a particular profile, pass `--profile <PROFILE-NAME>` to any Cargo command. The `dev`
profile is used by default, and the `release` profile can also be selected with `--release`.

- The `dev` profile (default) enables fast incremental compilation. It is useful for the usual
  modify-compile-run cycle of software develompent.
- The `lto` profile is like `release`, but has `lto="thin"`. This option provides some performance gains
  at the cost of a few extra seconds of compile time.
- The `release` profile, based on Cargo's default release profile, sets `lto=true`. This option adds
  a lot of compilation time. It's unclear how significant the performance difference
  from `lto="thin"` is, but it's certainly not very obvious.

When running `csl-perf` and comparing results, ensure you are using the same profile and compilation
settings. The `lto` profile is likely sufficient for this particular use case.

Further reading: [The Cargo Book, "3.5 Profiles," section on LTO](https://doc.rust-lang.org/cargo/reference/profiles.html#lto).
