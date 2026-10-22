import os

import pytest

from bistmas.data.clients.http import PoliteSession
from bistmas.shared.config import load_settings


@pytest.fixture(scope="session")
def settings():
    return load_settings()


def pytest_collection_modifyitems(config, items):
    if os.environ.get("BISTMAS_NETWORK") == "1":
        return
    skip = pytest.mark.skip(reason="network tests disabled (set BISTMAS_NETWORK=1)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    """Unit tests never touch the network: PoliteSession.request raises unless the test is @network."""
    if "network" in request.keywords or os.environ.get("BISTMAS_NETWORK") == "1":
        return

    def _blocked(self, method, url, **kwargs):  # noqa: ANN001
        raise RuntimeError(f"network access blocked in unit tests: {method} {url}")

    monkeypatch.setattr(PoliteSession, "request", _blocked)
