import queue
import threading
from typing import Any, NamedTuple


class ThreadedIterator(object):
    """An iterator that computes its elements in a parallel thread to be ready to
    be consumed.
    Exceptions raised by the threaded iterator are propagated to consumer.
    """

    def __init__(
        self,
        iterator,
        max_queue_size: int = 2,
        start: bool = True,
    ):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=lambda: self.worker(iterator))
        if start:
            self.Start()

    def Start(self):
        self._thread.start()

    def worker(self, iterator):
        try:
            for element in iterator:
                self._queue.put(self._ValueOrError(value=element), block=True)
        except Exception as e:
            # Propagate an error in the iterator.
            self._queue.put(self._ValueOrError(error=e))
        # Mark that the iterator is done.
        self._queue.put(self._EndOfIterator(), block=True)

    def __iter__(self):
        next_element = self._queue.get(block=True)
        while not isinstance(next_element, self._EndOfIterator):
            value = next_element.GetOrRaise()
            yield value
            next_element = self._queue.get(block=True)
        self._thread.join()

    class _EndOfIterator(object):
        """Tombstone marker object for iterators."""

        pass

    class _ValueOrError(NamedTuple):
        """A tuple which represents the union of either a value or an error."""

        value: Any = None
        error: Exception = None

        def GetOrRaise(self) -> Any:
            """Return the value or raise the exception."""
            if self.error is None:
                return self.value
            else:
                raise self.error
