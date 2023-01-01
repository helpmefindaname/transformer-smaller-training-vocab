import json
from pathlib import Path

import pytest


@pytest.fixture
def testdata():
    base_path = Path(__file__).parent / "tests" / "data"
    return base_path.joinpath


@pytest.fixture
def test_snapshots(gen_snapshots):
    def inner(output, output_path: Path):

        if gen_snapshots:
            with output_path.open("w+", encoding="utf-8") as f:
                json.dump(output, f, indent=4, sort_keys=True, ensure_ascii=False)
        else:
            with output_path.open("r", encoding="utf-8") as f:
                expected = json.load(f)
            assert output == expected

    return inner


def pytest_addoption(parser):
    parser.addoption("--gen-snapshots", action="store_true", help="recreates outputs instead of testing")


def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.gen_snapshots
    if "gen_snapshots" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("gen_snapshots", [option_value])
