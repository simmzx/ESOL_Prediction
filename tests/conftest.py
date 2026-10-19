import os
from pathlib import Path
import pytest

@pytest.fixture
def data_path():
    return os.environ.get('ESOL_CSV','/home/zhangxiang_cluster/assign1/esol.csv')
