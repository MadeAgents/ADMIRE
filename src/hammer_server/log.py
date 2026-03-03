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

import logging
import os


def setup_logger(level=None, file_info: str = None, file_err: str = None):
    if level is None:
        if os.getenv("HAMMER_ENV_DEBUG", "0").strip().lower() in ("1", "true"):
            level = logging.DEBUG
        else:
            level = logging.INFO

    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_info_handler = None
    if file_info:
        file_info_handler = logging.FileHandler(file_info)
        file_info_handler.setFormatter(formatter)
        file_info_handler.setLevel(logging.INFO)

    file_err_handler = None
    if file_err:
        file_err_handler = logging.FileHandler(file_err)
        file_err_handler.setFormatter(formatter)
        file_err_handler.setLevel(logging.ERROR)

    _logger = logging.getLogger()
    # _logger.setLevel(logging.NOTSET)
    _logger.addHandler(console_handler)
    if file_info_handler:
        _logger.addHandler(file_info_handler)
    if file_err_handler:
        _logger.addHandler(file_err_handler)
    return _logger


# logger = setup_logger()
