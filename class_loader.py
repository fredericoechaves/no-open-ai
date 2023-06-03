import importlib

class Loader:
	def load(self, module="", class_name=""):
		mod = importlib.import_module(module)
		return getattr(mod, class_name)
