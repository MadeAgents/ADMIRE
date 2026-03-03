# Copyright 2026 OPPO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager

import logging
import time

logger = logging.getLogger(__file__)


@contextmanager
def timer(name: str, log: dict = None):
    """
    Context-manager that records elapsed time (in seconds) under `name`
    into the provided dictionary `log`.

    Usage:
        >>> timings = {}
        >>> with timer("task_A", timings):
        ...     do_something()
        >>> print(timings)   # {'task_A': 0.123}
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.debug(f"[{name}] took {elapsed:.3f} s")
        if log is not None:
            log[name] = elapsed


if __name__ == "__main__":
    timings = {}

    with timer("sleep_1s", timings):
        time.sleep(1)

    with timer("sleep_half", timings):
        time.sleep(0.5)

    print(timings)  # {'sleep_1s': 1.001, 'sleep_half': 0.501}
