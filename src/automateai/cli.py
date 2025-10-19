import sys
from typing import Optional

import typer

from . import __version__


app = typer.Typer(help="automateai command-line interface")


@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


def _launch_streamlit_app() -> None:
    try:
        # Import Streamlit's CLI entry
        from streamlit.web.cli import main as st_main  # type: ignore
    except Exception:  # pragma: no cover - only triggered when streamlit isn't installed
        typer.echo(
            "Streamlit is not installed. Install app extras: `pip install automateai[app]`",
            err=True,
        )
        raise typer.Exit(code=1)

    # Execute: streamlit run -m automateai.app
    sys.argv = ["streamlit", "run", "-m", "automateai.app"]
    st_main()


@app.command(name="app")
def run_app() -> None:
    """Launch the Streamlit app."""
    _launch_streamlit_app()


def main(argv: Optional[list[str]] = None) -> None:
    """Entrypoint for console_scripts."""
    app()


