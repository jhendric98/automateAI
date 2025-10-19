def test_package_import() -> None:
    import automateai  # noqa: F401


def test_version_present() -> None:
    import automateai

    assert isinstance(automateai.__version__, str)
    assert automateai.__version__ != ""


