# OpenWolf

@.wolf/OPENWOLF.md

This project uses OpenWolf for context management. Read and follow .wolf/OPENWOLF.md every session. Check .wolf/cerebrum.md before generating code. Check .wolf/anatomy.md before reading files.

## Fast Rust Rebuilds

Source `scripts/dev-env.sh` to activate sccache and a shared target dir:

```sh
source scripts/dev-env.sh
```

Sets `RUSTC_WRAPPER=sccache`, `CARGO_INCREMENTAL=1`, and `CARGO_TARGET_DIR=$HOME/.cargo-target-shared`.
Does not modify shell config files; re-sourcing is safe (idempotent).
