# performance_monitor.py
import time
import threading
import subprocess
import psutil
import os

class PerformanceMonitor:
    def __init__(self, interval=10):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())

        self.cpu_util_samples = []
        self.mem_samples = []
        self.gpu_util_samples = []
        self.gpu_mem_samples = []
        self.gpu_power_samples = []

        self._running = False
        self._start_time = None
        self._end_time = None

    def get_gpu_stat(self, query):
        try:
            result = subprocess.run(
                ['nvidia-smi', f'--query-gpu={query}', '--format=csv,nounits,noheader'],
                stdout=subprocess.PIPE, text=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            print(f"[Monitor Warning] Failed to get {query}: {e}")
            return 0.0

    def monitor(self):
        # Prime CPU usage
        self.proc.cpu_percent(interval=None)

        while self._running:
            try:
                self.cpu_util_samples.append(self.proc.cpu_percent(interval=None))
                mem = self.proc.memory_info().rss / (1024 ** 3)  # in GB
                self.mem_samples.append(mem)
                self.gpu_util_samples.append(self.get_gpu_stat("utilization.gpu"))
                self.gpu_mem_samples.append(self.get_gpu_stat("memory.used") / 1024)  # MB -> GB
                self.gpu_power_samples.append(self.get_gpu_stat("power.draw"))
            except Exception as e:
                print(f"[Monitor Error] {e}")
            time.sleep(self.interval)

    def start(self):
        print("[Monitor] Starting performance monitor...")
        self._running = True
        self._start_time = time.time()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self._running = False
        self._end_time = time.time()
        self.thread.join()

    def report(self):
        if self._end_time is None:
            self._end_time = time.time()

        runtime = self._end_time - self._start_time
        avg_cpu = sum(self.cpu_util_samples) / len(self.cpu_util_samples) if self.cpu_util_samples else 0
        avg_mem = sum(self.mem_samples) / len(self.mem_samples) if self.mem_samples else 0
        avg_gpu_util = sum(self.gpu_util_samples) / len(self.gpu_util_samples) if self.gpu_util_samples else 0
        avg_gpu_mem = sum(self.gpu_mem_samples) / len(self.gpu_mem_samples) if self.gpu_mem_samples else 0
        avg_gpu_power = sum(self.gpu_power_samples) / len(self.gpu_power_samples) if self.gpu_power_samples else 0

        #cpu_energy_joules = self.tdp * (avg_cpu / 100) * runtime
        ram_energy_joules = 0.375 * avg_mem * runtime
        gpu_energy_joules = avg_gpu_power * runtime

        print("\n====== Performance Summary ======")
        print(f"Total Runtime           : {runtime:.2f} sec")
        print(f"Avg CPU Utilization     : {avg_cpu:.2f}%")
        print(f"CPU Load Time           : {(avg_cpu / 100) * runtime:.2f} sec")
        #print(f"Estimated CPU Energy    : {cpu_energy_joules:.2f} J")
        print(f"Avg RAM Usage           : {avg_mem:.2f} GB")
        print(f"Estimated RAM Energy    : {ram_energy_joules:.2f} J")
        print(f"Avg GPU Utilization     : {avg_gpu_util:.2f}%")
        print(f"Avg GPU Memory Usage    : {avg_gpu_mem:.2f} GB")
        print(f"Avg GPU Power Draw      : {avg_gpu_power:.2f} W")
        print(f"Estimated GPU Energy    : {gpu_energy_joules:.2f} J")
        print("=================================\n")

