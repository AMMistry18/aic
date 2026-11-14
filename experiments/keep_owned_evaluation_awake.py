"""Hold an idle-sleep request only while a named existing evaluator PID lives."""
import argparse
import ctypes
from ctypes import wintypes

p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);a=p.parse_args()
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
kernel.OpenProcess.restype=wintypes.HANDLE
handle=kernel.OpenProcess(0x100000,False,a.pid)
if not handle:raise ctypes.WinError(ctypes.get_last_error())
kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
kernel.CloseHandle.argtypes=[wintypes.HANDLE]
requested=kernel.SetThreadExecutionState(0x80000001)
if not requested:raise ctypes.WinError(ctypes.get_last_error())
print(f'Idle sleep request active for evaluator PID {a.pid}; no display or power-plan changes',flush=True)
try:
    while True:
        status=kernel.WaitForSingleObject(handle,1000)
        if status==0:break
        if status!=258:raise ctypes.WinError(ctypes.get_last_error())
finally:
    kernel.SetThreadExecutionState(0x80000000);kernel.CloseHandle(handle)
    print('Evaluator ended; idle sleep request released',flush=True)
