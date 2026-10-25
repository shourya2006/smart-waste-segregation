class _ComponentStub:
    """Stub for any Gradio component returned inside a Blocks context."""

    def __init__(self, *args, **kwargs):
        pass

    def click(self, *args, **kwargs):
        return self

    def change(self, *args, **kwargs):
        return self


class Blocks:
    """A minimal stub of the gradio.Blocks class for testing purposes.
    Provides a no-op ``launch`` method so that tests can patch it without
    requiring the real Gradio library.
    Accepts arbitrary keyword arguments (e.g. css=) so that the real
    ``build_interface`` signature works without the full library installed.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    @staticmethod
    def launch(*args, **kwargs):  # pragma: no cover
        """Placeholder launch method.
        The real Gradio ``Blocks.launch`` starts a web server; the stub does nothing.
        """
        return None


# ---------------------------------------------------------------------------
# Stub component factories – return no-op objects so that build_interface()
# can be called in the test environment without the full Gradio library.
# ---------------------------------------------------------------------------

def Markdown(*args, **kwargs):
    return _ComponentStub()


def Image(*args, **kwargs):
    return _ComponentStub()


def Button(*args, **kwargs):
    return _ComponentStub()


def Textbox(*args, **kwargs):
    return _ComponentStub()


def Dataframe(*args, **kwargs):
    return _ComponentStub()


class Row:
    """Stub for gr.Row() used as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
