# tests/test_preprocessing.py
import pytest
import numpy as np
from src.data_preprocessing import ip_to_int

def test_valid_ip_conversion():
    assert ip_to_int('192.168.1.1') == 3232235777.0

def test_null_ip_handling():
    assert np.isnan(ip_to_int(np.nan))