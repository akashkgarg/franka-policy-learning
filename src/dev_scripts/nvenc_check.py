import ctypes
lib = ctypes.CDLL("libnvidia-encode.so.1")
v = ctypes.c_uint32()
print("NvEncodeAPIGetMaxSupportedVersion:", lib.NvEncodeAPIGetMaxSupportedVersion(ctypes.byref(v)), v.value)
