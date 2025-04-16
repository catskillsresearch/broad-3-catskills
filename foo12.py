import cuml
from cuml.common.device_selection import using_device_type

print(f"cuML version: {cuml.__version__}")
print(f"Active GPU ID: {cuml.get_device_id()}")

# Verify GPU usage context
with using_device_type('gpu'):
    print("GPU acceleration is enabled")
