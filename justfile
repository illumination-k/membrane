# Format code (Rust + dprint)
fmt:
    cargo fmt --all
    dprint fmt

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check
    dprint check

# Run clippy
lint:
    cargo clippy --all-targets -- -D warnings

# Run all checks (fmt + lint)
check: fmt-check lint

# Run tests
test:
    cargo test --all

# Build
build:
    cargo build --all
