"""Stdlib-only lifecycle regressions for the actual batching module."""

import importlib.util
import sys
import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "optillm_batching_lifecycle", Path(__file__).resolve().parents[1] / "optillm" / "batching.py"
)
batching = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batching
SPEC.loader.exec_module(batching)


class BatchingLifecycleTests(unittest.TestCase):
    def submit(self, batcher, text, entered=None):
        result = Future()

        def run():
            if entered is not None:
                entered.set()
            try:
                result.set_result(batcher.add_request({"model": "test", "prompt": text}))
            except BaseException as error:
                result.set_exception(error)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        return result

    def make_batcher(self):
        batcher = batching.RequestBatcher(max_batch_size=1, max_wait_ms=10, enable_logging=False)
        batcher.set_processor(lambda requests: [{"text": item["prompt"]} for item in requests])
        self.addCleanup(batcher.shutdown)
        return batcher

    def test_submissions_after_shutdown_are_rejected_without_waiting(self):
        batcher = self.make_batcher()
        batcher.shutdown()
        batcher.shutdown()
        result = self.submit(batcher, "after shutdown")
        with self.assertRaises(batching.BatchingError):
            result.result(timeout=1)
        self.assertEqual(batcher.queues, {})

    def test_queued_request_fails_while_inflight_request_can_finish(self):
        batcher = self.make_batcher()
        started = threading.Event()
        release = threading.Event()
        queued = threading.Event()

        def processor(requests):
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError("test processor release timed out")
            return [{"text": item["prompt"]} for item in requests]

        batcher.set_processor(processor)
        first = self.submit(batcher, "inflight")
        self.assertTrue(started.wait(timeout=1))
        queue_obj = next(iter(batcher.queues.values()))
        original_put = queue_obj.put

        def put(item, *args, **kwargs):
            original_put(item, *args, **kwargs)
            if isinstance(item, batching.BatchRequest):
                queued.set()

        closer = None
        try:
            with patch.object(queue_obj, "put", side_effect=put):
                second = self.submit(batcher, "queued")
                self.assertTrue(queued.wait(timeout=1))
                closer = threading.Thread(target=batcher.shutdown, daemon=True)
                closer.start()
                with self.assertRaises(batching.BatchingError):
                    second.result(timeout=1)
        finally:
            release.set()
            if closer is not None:
                closer.join(timeout=2)
        self.assertEqual(first.result(timeout=1), {"text": "inflight"})

    def test_shutdown_during_queue_registration_does_not_strand_the_caller(self):
        batcher = self.make_batcher()
        entered = threading.Event()
        release = threading.Event()
        closed = threading.Event()
        original_queue = batching.queue.Queue

        def create_queue(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("test queue release timed out")
            return original_queue(*args, **kwargs)

        def close():
            batcher.shutdown()
            closed.set()

        closer = None
        try:
            with patch.object(batching.queue, "Queue", side_effect=create_queue):
                result = self.submit(batcher, "registered concurrently")
                self.assertTrue(entered.wait(timeout=1))
                closer = threading.Thread(target=close, daemon=True)
                closer.start()
                # The original implementation returns here before registering
                # the queue. The fixed implementation waits for admission.
                closed.wait(timeout=0.1)
                release.set()
                try:
                    response = result.result(timeout=1)
                except batching.BatchingError:
                    pass
                else:
                    self.assertEqual(response, {"text": "registered concurrently"})
        finally:
            release.set()
            if closer is not None:
                closer.join(timeout=2)
        self.assertTrue(closed.is_set())

    def test_concurrent_initial_requests_share_one_queue(self):
        batcher = self.make_batcher()
        entered = threading.Event()
        release = threading.Event()
        second_constructor = threading.Event()
        second_entered = threading.Event()
        original_queue = batching.queue.Queue
        constructed = []

        def create_queue(*args, **kwargs):
            constructed.append(True)
            if len(constructed) == 1:
                entered.set()
                if not release.wait(timeout=3):
                    raise RuntimeError("test queue release timed out")
            else:
                second_constructor.set()
            return original_queue(*args, **kwargs)

        try:
            with patch.object(batching.queue, "Queue", side_effect=create_queue):
                first = self.submit(batcher, "first")
                self.assertTrue(entered.wait(timeout=1))
                second = self.submit(batcher, "second", entered=second_entered)
                self.assertTrue(second_entered.wait(timeout=1))
                second_constructor.wait(timeout=0.1)
                release.set()
                self.assertEqual(first.result(timeout=1), {"text": "first"})
                self.assertEqual(second.result(timeout=1), {"text": "second"})
        finally:
            release.set()
        self.assertEqual(len(constructed), 1)

    def test_shutdown_wakes_an_idle_worker_with_a_long_batch_deadline(self):
        batcher = batching.RequestBatcher(max_batch_size=1, max_wait_ms=60000, enable_logging=False)
        batcher.set_processor(lambda requests: [{"text": item["prompt"]} for item in requests])
        self.addCleanup(batcher.shutdown)
        idle = threading.Event()
        original_queue = batching.queue.Queue

        class ObservedQueue(original_queue):
            calls = 0

            def get(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    idle.set()
                return super().get(*args, **kwargs)

        with patch.object(batching.queue, "Queue", ObservedQueue):
            result = self.submit(batcher, "before idle")
            self.assertEqual(result.result(timeout=1), {"text": "before idle"})
            self.assertTrue(idle.wait(timeout=1))
            batcher.shutdown()
        self.assertTrue(all(not thread.is_alive() for thread in batcher.batch_threads.values()))

    def test_processor_can_request_shutdown_without_joining_itself(self):
        batcher = self.make_batcher()

        def processor(requests):
            batcher.shutdown()
            return [{"text": item["prompt"]} for item in requests]

        batcher.set_processor(processor)
        result = self.submit(batcher, "last response")
        self.assertEqual(result.result(timeout=1), {"text": "last response"})


if __name__ == "__main__":
    unittest.main()
