from importlib.metadata import PackageNotFoundError, version


def _get_version() -> str:
    try:
        return version("automateai")
    except PackageNotFoundError:
        # Fallback for editable installs or when metadata isn't available
        return "0.0.0"


__version__ = _get_version()

__all__ = ["__version__"]


