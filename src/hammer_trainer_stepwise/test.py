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

import pynvml
import time
import threading
import torch

pynvml.nvmlInit()
num_gpus = 8
handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]

def get_gpu_utilization(gpu_index):
    util = pynvml.nvmlDeviceGetUtilizationRates(handles[gpu_index])
    return util.gpu  # 利用率百分比

class GpuLoadThread(threading.Thread):
    def __init__(self, gpu_index):
        super().__init__()
        self.gpu_index = gpu_index
        self.running = threading.Event()
        self.running.clear()  # 初始不运行
        self.stop_signal = threading.Event()
        self.device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

    def run(self):
        while not self.stop_signal.is_set():
            if self.running.is_set():
                start_time = time.time()
                while time.time() - start_time < 9:
                    a = torch.rand((1200, 1200), device=self.device)
                    b = torch.rand((1200, 1200), device=self.device)
                    c = torch.mm(a, b)
                    torch.cuda.synchronize()
                time.sleep(1)  # 9秒计算+1秒休息
            else:
                time.sleep(0.1)

    def pause(self):
        self.running.clear()

    def resume(self):
        self.running.set()

    def stop(self):
        self.stop_signal.set()
        self.running.set()


def main():
    threads = [GpuLoadThread(i) for i in range(num_gpus)]
    for t in threads:
        t.start()

    try:
        while True:
            for i in range(num_gpus):
                gpu_util = get_gpu_utilization(i)
                print(f"GPU-{i} 利用率: {gpu_util}% 线程状态: {'运行' if threads[i].running.is_set() else '暂停'}")

                if gpu_util == 0 and not threads[i].running.is_set():
                    print(f"GPU-{i} 空闲，启动负载线程")
                    threads[i].resume()

                elif gpu_util >= 90 and threads[i].running.is_set():
                    print(f"GPU-{i} 利用率高，暂停负载线程")
                    threads[i].pause()

            time.sleep(2)  # 每2秒检测一次

    except KeyboardInterrupt:
        print("程序终止，停止所有负载线程...")
        for t in threads:
            t.stop()
        for t in threads:
            t.join()

    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()