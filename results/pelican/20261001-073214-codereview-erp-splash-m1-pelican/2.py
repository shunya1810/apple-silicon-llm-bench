dropped += 1
self._queue.task_done()      # <- no _release_pending(pending.*) here
