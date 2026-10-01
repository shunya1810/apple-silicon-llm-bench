estimated_nbytes = int(getattr(entry, "nbytes", 0) or 0)
if not self._admit_write(estimated_nbytes):   # _pending_bytes += estimated
    return False
...
pending = PendingWrite(..., pinned_nbytes=max(estimated_nbytes, int(encoded.nbytes)))
