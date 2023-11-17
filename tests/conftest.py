from os import path

import pytest

TEST_DIR = path.dirname(__file__)


@pytest.fixture
def test_data_dir():
    return path.join(TEST_DIR, "test_data")
