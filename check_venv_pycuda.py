import importlib, sys
try:
    pycuda = importlib.import_module('pycuda')
    print('pycuda VERSION:', getattr(pycuda, 'VERSION', None))
    drv = importlib.import_module('pycuda.driver')
    print('pycuda driver.get_version():', drv.get_version())
except Exception as e:
    print('pycuda import error:', e)
    sys.exit(0)
