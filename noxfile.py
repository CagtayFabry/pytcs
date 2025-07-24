from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()
PROJECT = nox.project.load_toml()

nox.needs_version = ">=2025.2.9"
nox.options.sessions = ["lint", "tests"]
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    test_deps = nox.project.dependency_groups(PROJECT, "test")
    session.install("-e .", *test_deps)
    session.run("uv", "pip", "list")
    session.run("pytest", *session.posargs)


# @nox.session(python=["3.10"])
# @nox.parametrize("numpy,pandas", [("1.*", "2.*"), ("2.*", "2.*")])
# def tests_compat(session, pandas, numpy):
#     session.install(".[test]", f"pandas=={pandas}", f"numpy=={numpy}")
#     session.run("uv", "pip", "list")
#     session.run("pytest", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass --non-interactive to avoid serving.

    First positional argument is the target directory.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    parser.add_argument("output", nargs="?", help="Output directory")
    args, posargs = parser.parse_known_args(session.posargs)
    serve = args.builder == "html" and session.interactive

    session.install("-e.[docs]", "sphinx-autobuild")

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        args.output or f"docs/_build/{args.builder}",
        *posargs,
    )

    if serve:
        session.run("sphinx-autobuild", "--open-browser", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session
def build_api_docs(session: nox.Session) -> None:
    """
    Build (regenerate) API docs.
    """

    session.install("sphinx")
    session.run(
        "sphinx-apidoc",
        "-o",
        "docs/api/",
        "--module-first",
        "--no-toc",
        "--force",
        "src/pylibo",
    )


@nox.session
def build(session: nox.Session) -> None:
    """
    Build an SDist and wheel.
    """

    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")


def _get_session_config(arg) -> list[str]:
    """Get session configuration list of strings."""
    session_func = globals()[arg]

    # list all sessions for this base session
    try:
        session_func.parametrize  # noqa: B018
    except AttributeError:
        sessions_list = [f"{session_func.__name__}-{py}" for py in session_func.python]
    else:
        sessions_list = [
            f"{session_func.__name__}-{py}({param})"
            for py, param in itertools.product(
                session_func.python, session_func.parametrize
            )
        ]

    return sessions_list


@nox.session(python=False)
def gha_list(session):
    """Prints all sessions available for <base_session_name>, for GithubActions.

    (mandatory arg: <base_session_name>)

    source: https://stackoverflow.com/a/66747360/11242411
    """

    # get the desired base session to generate the list for
    if len(session.posargs) < 1:
        raise ValueError("This session has a mandatory argument: <base_session_name>")

    sessions_list = [_get_session_config(arg) for arg in session.posargs]

    sessions_list = list(itertools.chain.from_iterable(sessions_list))

    # print the list so that it can be caught by GHA.
    # Note that json.dumps is optional since this is a list of string.
    # However it is to remind us that GHA expects a well-formatted json list of strings.
    print(json.dumps(sessions_list))  # noqa: T201
