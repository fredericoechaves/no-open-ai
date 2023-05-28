import time
import sys

class Profile:
	def __init__(self):
		self.start_time = time.process_time()

	def log(self, message=""):
		print(f"{message}{time.process_time() - self.start_time :.2f}", file=sys.stderr)
		self.start_time = time.process_time()
