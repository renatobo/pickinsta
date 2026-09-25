def _run() -> None:
    """Load the CLI only when the package is executed as a module."""
    from pickinsta.cli import main

    main()


if __name__ == "__main__":
    _run()
