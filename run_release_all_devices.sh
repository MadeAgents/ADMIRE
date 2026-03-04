#!/bin/bash

url=""

python -c "from client import HammerEnvClient; client = HammerEnvClient(\"${url}\"); client.release_all_devices()"
