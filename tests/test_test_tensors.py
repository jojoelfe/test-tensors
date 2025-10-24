import test_tensors


def test_imports_with_version():
    assert isinstance(test_tensors.__version__, str)
