import pytest

from filings_lab.synthetic import SyntheticConfig, generate


@pytest.fixture(scope="session")
def small_world(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    world = generate(SyntheticConfig(n_companies=6, start="2016-01-01", end="2020-12-31", seed=11), out)
    world["dir"] = out
    return world


@pytest.fixture(scope="session")
def small_lake(small_world, tmp_path_factory):
    from filings_lab.pipeline import local

    lake = tmp_path_factory.mktemp("lake")
    local.run(small_world["index"], small_world["prices"], str(lake), workers=1)
    return lake
