def pytest_generate_tests(metafunc):
    if "dataset" in metafunc.fixturenames:
        metafunc.parametrize("dataset", metafunc.cls.datasets)
