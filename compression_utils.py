from src.medifederate.federated import pack,unpack

def quantize_state(state_dict,bits=8):
    if bits != 8: raise ValueError('This consolidated implementation currently supports 8-bit quantization.')
    return pack(state_dict,bits)[0]

def dequantize_state(packed): return unpack(packed)
