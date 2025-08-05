"""
Pure C implementation benchmark using ctypes.
"""

import ctypes
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn

console = Console()


class CBenchmark:
    """Benchmark pure C StableswapMath implementation with GMP."""
    
    def __init__(self, lib_path: Path):
        self.lib_path = lib_path
        self.lib = None
        self.load_library()
    
    def load_library(self):
        """Load the C shared library."""
        console.print(f"[yellow]Loading C library from {self.lib_path}...[/yellow]")
        
        if not self.lib_path.exists():
            raise FileNotFoundError(f"C library not found at {self.lib_path}")
        
        self.lib = ctypes.CDLL(str(self.lib_path))
        
        # Initialize math library
        self.lib.stableswap_math_init()
        
        # Set up function signatures
        # newton_D
        self.lib.newton_D.argtypes = [ctypes.c_char_p, ctypes.c_char_p, 
                                      ctypes.c_char_p, ctypes.c_char_p]
        self.lib.newton_D.restype = ctypes.c_char_p
        
        # get_y
        self.lib.get_y.argtypes = [ctypes.c_char_p, ctypes.c_char_p, 
                                   ctypes.c_char_p, ctypes.c_char_p,
                                   ctypes.c_char_p, ctypes.c_int]
        self.lib.get_y.restype = ctypes.POINTER(ctypes.c_char_p)
        
        # get_p
        self.lib.get_p.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                   ctypes.c_char_p, ctypes.c_char_p]
        self.lib.get_p.restype = ctypes.c_char_p
        
        # wad_exp
        self.lib.wad_exp.argtypes = [ctypes.c_char_p]
        self.lib.wad_exp.restype = ctypes.c_char_p
        
        # free_string_array
        self.lib.free_string_array.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
        
        console.print("[green]✓ C library loaded[/green]")
    
    def newton_D(self, A: str, gamma: str, x0: str, x1: str) -> str:
        """Call newton_D function."""
        result = self.lib.newton_D(
            A.encode('utf-8'),
            gamma.encode('utf-8'),
            x0.encode('utf-8'),
            x1.encode('utf-8')
        )
        result_str = result.decode('utf-8')
        # Free the C string
        self.lib.free(result)
        return result_str
    
    def get_y(self, A: str, gamma: str, x0: str, x1: str, D: str, i: int) -> tuple[str, str]:
        """Call get_y function."""
        result_ptr = self.lib.get_y(
            A.encode('utf-8'),
            gamma.encode('utf-8'),
            x0.encode('utf-8'),
            x1.encode('utf-8'),
            D.encode('utf-8'),
            i
        )
        
        if not result_ptr:
            raise RuntimeError("get_y returned NULL")
        
        # Extract results
        y = result_ptr[0].decode('utf-8')
        k0_prev = result_ptr[1].decode('utf-8')
        
        # Free the array
        self.lib.free_string_array(result_ptr, 2)
        
        return y, k0_prev
    
    def get_p(self, x0: str, x1: str, D: str, A: str) -> str:
        """Call get_p function."""
        result = self.lib.get_p(
            x0.encode('utf-8'),
            x1.encode('utf-8'),
            D.encode('utf-8'),
            A.encode('utf-8')
        )
        result_str = result.decode('utf-8')
        self.lib.free(result)
        return result_str
    
    def wad_exp(self, x: str) -> str:
        """Call wad_exp function."""
        result = self.lib.wad_exp(x.encode('utf-8'))
        if not result:
            raise RuntimeError("wad_exp overflow")
        result_str = result.decode('utf-8')
        self.lib.free(result)
        return result_str
    
    def __del__(self):
        """Cleanup when done."""
        if self.lib:
            self.lib.stableswap_math_cleanup()


# For compatibility with existing benchmark code
CppBenchmark = CBenchmark