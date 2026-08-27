# Exact audited process-tree RSS monitor.
import os
import threading
import time

import psutil


class ProcessTreeRSSMonitor:
    """Sample the benchmark process and every descendant, including R."""

    def __init__(self, interval_seconds=0.05):
        self.interval = float(interval_seconds)
        self.process = psutil.Process(os.getpid())
        self.initial_mb = self._rss_mb()
        self.peak_mb = self.initial_mb
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _rss_mb(self):
        total = 0
        processes = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        for process in processes:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total / (1024.0 ** 2)

    def _sample(self):
        while not self._stop.wait(self.interval):
            self.peak_mb = max(self.peak_mb, self._rss_mb())

    def __enter__(self):
        self._thread.start()
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval * 4))
        self.peak_mb = max(self.peak_mb, self._rss_mb())
        self.elapsed_seconds = time.perf_counter() - self.started_at
        self.increase_mb = max(0.0, self.peak_mb - self.initial_mb)
