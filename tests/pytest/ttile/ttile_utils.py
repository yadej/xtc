
def requires_tvm(*arg):
    import pytest
    def has_tvm():
        try:
            import tvm
            return True
        except:
            return False
    return pytest.mark.skipif(not has_tvm(), reason="requires TVM")(*arg)
