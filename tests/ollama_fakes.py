"""Stand-ins for what ``requests.post`` returns, shared by the tests that fake Ollama's HTTP."""

import json


class FakeResponse:
    """The parts of ``requests.Response`` that ``OllamaClient`` uses.

    ``payload`` is the JSON body of a plain reply. ``lines`` (any iterable, possibly a generator
    that logs when it is read, or one that raises) is the body of a streamed reply. ``closed``
    tells whether the client released the connection.
    """

    def __init__(self, status_code=200, payload=None, text=None, lines=None):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload
        self._lines = lines
        self.text = text if text is not None else json.dumps(payload)
        self.closed = False

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload

    def iter_lines(self):
        yield from self._lines if self._lines is not None else ()

    def close(self):
        self.closed = True


def chat_reply(message):
    """A plain (not streamed) reply holding one Ollama chat message."""
    return FakeResponse(200, {"message": message})


def stream_reply(lines):
    """A streamed reply made of the given NDJSON lines."""
    return FakeResponse(200, lines=lines)


def line(**fields):
    """One NDJSON line of a streamed reply."""
    return json.dumps(fields).encode()


def piece(content, **message):
    """A streamed chunk carrying a piece of the answer."""
    return line(model="m", message={"role": "assistant", "content": content, **message}, done=False)


def finished(**extra):
    """The last chunk of a streamed reply."""
    message = {"role": "assistant", "content": ""}
    return line(model="m", message=message, done=True, done_reason="stop", **extra)
