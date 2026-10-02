import textwrap

import pytest

from dspo.graph_builder import build_graph

FILES = {
    "pkg/__init__.py": """
        from .client import Client
        from .utils import helper as public_helper
    """,
    "pkg/utils.py": """
        def helper(x):
            return parse_value(x)


        def parse_value(x):
            return int(x)


        def very_unique_function_name():
            return 1
    """,
    "pkg/base.py": """
        class Base:
            def close(self):
                return None

            def finish(self):
                return 1
    """,
    "pkg/client.py": """
        import pkg.utils
        from . import utils
        from .base import Base
        from .utils import helper


        class Client(Base):
            def __init__(self, url):
                self.url = url

            def send(self, request):
                data = helper(request)
                self.prepare(data)
                self.finish()
                return utils.parse_value(data)

            def prepare(self, data):
                obj = make_obj()
                obj.very_unique_function_name()
                obj.update({})
                return pkg.utils.parse_value(data)


        def make_obj():
            return Client("x")
    """,
    "tests/test_client.py": """
        def test_send():
            pass
    """,
}


@pytest.fixture
def repo(tmp_path):
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip())
    return tmp_path


@pytest.fixture
def graph(repo):
    return build_graph(repo)
