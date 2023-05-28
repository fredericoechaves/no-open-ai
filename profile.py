import time
import sys

class Profile:
	def __init__(self):
		self.start_time = time.time()

	def log(self, message=""):
		print(f"{message}{time.time() - self.start_time :.2f}", file=sys.stderr)
		self.start_time = time.time()
