from embedding_as_service.utils import any2unicode

def test_any2unicode():
    assert any2unicode("hello") == "hello"
    assert any2unicode(b"hello") == "hello"
