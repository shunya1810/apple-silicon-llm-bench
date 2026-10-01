dropped += 1
self._queue.task_done()          # reservation for this item is never released
