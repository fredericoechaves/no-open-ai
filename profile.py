import time
import sys

class Profile:
	def __init__(self):
		self.start_time = time.perf_counter()

	def log(self, message=""):
		print(f"{message}({time.perf_counter() - self.start_time :.2f})", file=sys.stderr)
		self.start_time = time.perf_counter()
