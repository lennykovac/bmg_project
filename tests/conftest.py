import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--exhaustive",
        action="store_true",
        default=False,
        help="run compute-intensive tests marked with @pytest.mark.exhaustive",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--exhaustive"):
        return
    skip_exhaustive = pytest.mark.skip(reason="need --exhaustive option to run")
    for item in items:
        if "exhaustive" in item.keywords:
            item.add_marker(skip_exhaustive)
