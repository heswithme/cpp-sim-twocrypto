"""
Pure C implementation benchmark using ctypes.
"""

import ctypes
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn

console = Console()


class CppBenchmark:
    """Benchmark Pure C StableswapMath implementation with GMP."""
    
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
        self.lib.newton_D.restype = ctypes.POINTER(ctypes.c_char)
        
        # get_y
        self.lib.get_y.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int
        ]
        self.lib.get_y.restype = ctypes.POINTER(ctypes.c_char_p)
        
        # get_p
        self.lib.get_p.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_char_p
        ]
        self.lib.get_p.restype = ctypes.POINTER(ctypes.c_char)
        
        # wad_exp
        self.lib.wad_exp.argtypes = [ctypes.c_char_p]
        self.lib.wad_exp.restype = ctypes.POINTER(ctypes.c_char)
        
        # free_string_array
        self.lib.free_string_array.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
        
        # Also need standard C free for single strings
        self.libc = ctypes.CDLL(None)
        self.libc.free.argtypes = [ctypes.c_void_p]
        
        console.print("[green]✓ C library loaded[/green]")
    
    def _get_string_and_free(self, ptr: ctypes.POINTER(ctypes.c_char)) -> str:
        """Get string from C pointer and free it."""
        if not ptr:
            return None
        result = ctypes.cast(ptr, ctypes.c_char_p).value.decode('utf-8')
        self.libc.free(ptr)
        return result
    
    def benchmark_newton_D(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Benchmark newton_D function."""
        results = []
        times = []
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Benchmarking C newton_D...", total=len(test_cases))
            
            for case in test_cases:
                amp = case['A'].encode('utf-8')
                gamma = case.get('gamma', '145000000000000').encode('utf-8')
                x0 = case['x0'].encode('utf-8')
                x1 = case['x1'].encode('utf-8')
                
                # Warm-up call
                ptr = self.lib.newton_D(amp, gamma, x0, x1)
                _ = self._get_string_and_free(ptr)
                
                # Timed call
                start = time.perf_counter_ns()
                ptr = self.lib.newton_D(amp, gamma, x0, x1)
                result = self._get_string_and_free(ptr)
                end = time.perf_counter_ns()
                
                results.append(result)
                times.append(end - start)
                progress.update(task, advance=1)
        
        return {
            'results': results,
            'times': times,
            'avg_time_ns': sum(times) / len(times) if times else 0
        }
    
    def benchmark_get_y(self, test_cases: List[Dict], D_values: List[str]) -> Dict[str, Any]:
        """Benchmark get_y function."""
        results = []
        times = []
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Benchmarking C get_y...", total=len(test_cases))
            
            for i, (case, D) in enumerate(zip(test_cases, D_values)):
                if D == "0":
                    results.append("0")
                    times.append(0)
                    progress.update(task, advance=1)
                    continue
                
                amp = case['A'].encode('utf-8')
                gamma = case.get('gamma', '145000000000000').encode('utf-8')
                x0 = case['x0'].encode('utf-8')
                x1 = case['x1'].encode('utf-8')
                D_bytes = D.encode('utf-8')
                j = i % 2
                
                # Warm-up call
                ptr = self.lib.get_y(amp, gamma, x0, x1, D_bytes, j)
                if ptr:
                    self.lib.free_string_array(ptr, 2)
                
                # Timed call
                start = time.perf_counter_ns()
                ptr = self.lib.get_y(amp, gamma, x0, x1, D_bytes, j)
                if ptr:
                    result = ptr[0].decode('utf-8') if ptr[0] else "0"
                    self.lib.free_string_array(ptr, 2)
                else:
                    result = "0"
                end = time.perf_counter_ns()
                
                results.append(result)
                times.append(end - start)
                progress.update(task, advance=1)
        
        return {
            'results': results,
            'times': times,
            'avg_time_ns': sum(times) / len(times) if times else 0
        }
    
    def benchmark_get_p(self, test_cases: List[Dict], D_values: List[str]) -> Dict[str, Any]:
        """Benchmark get_p function."""
        results = []
        times = []
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Benchmarking C get_p...", total=len(test_cases))
            
            for case, D in zip(test_cases, D_values):
                if D == "0":
                    results.append("0")
                    times.append(0)
                    progress.update(task, advance=1)
                    continue
                
                x0 = case['x0'].encode('utf-8')
                x1 = case['x1'].encode('utf-8')
                D_bytes = D.encode('utf-8')
                amp = case['A'].encode('utf-8')
                
                # Warm-up call
                ptr = self.lib.get_p(x0, x1, D_bytes, amp)
                _ = self._get_string_and_free(ptr)
                
                # Timed call
                start = time.perf_counter_ns()
                ptr = self.lib.get_p(x0, x1, D_bytes, amp)
                result = self._get_string_and_free(ptr)
                end = time.perf_counter_ns()
                
                results.append(result)
                times.append(end - start)
                progress.update(task, advance=1)
        
        return {
            'results': results,
            'times': times,
            'avg_time_ns': sum(times) / len(times) if times else 0
        }
    
    def benchmark_wad_exp(self, test_cases: List[str]) -> Dict[str, Any]:
        """Benchmark wad_exp function."""
        results = []
        times = []
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Benchmarking C wad_exp...", total=len(test_cases))
            
            for x in test_cases:
                x_bytes = x.encode('utf-8')
                
                # Warm-up call
                try:
                    ptr = self.lib.wad_exp(x_bytes)
                    if ptr:
                        _ = self._get_string_and_free(ptr)
                except:
                    pass
                
                # Timed call
                start = time.perf_counter_ns()
                try:
                    ptr = self.lib.wad_exp(x_bytes)
                    if ptr:
                        result = self._get_string_and_free(ptr)
                    else:
                        result = "overflow"
                except:
                    result = "error"
                end = time.perf_counter_ns()
                
                results.append(result)
                times.append(end - start)
                progress.update(task, advance=1)
        
        return {
            'results': results,
            'times': times,
            'avg_time_ns': sum(times) / len(times) if times else 0
        }
    
    def newton_D(self, A: str, gamma: str, x0: str, x1: str) -> str:
        """Call newton_D function directly (for precision testing)."""
        ptr = self.lib.newton_D(
            A.encode('utf-8'),
            gamma.encode('utf-8'),
            x0.encode('utf-8'),
            x1.encode('utf-8')
        )
        return self._get_string_and_free(ptr)
    
    def get_y(self, A: str, gamma: str, x0: str, x1: str, D: str, i: int) -> tuple:
        """Call get_y function directly (for precision testing)."""
        ptr = self.lib.get_y(
            A.encode('utf-8'),
            gamma.encode('utf-8'),
            x0.encode('utf-8'),
            x1.encode('utf-8'),
            D.encode('utf-8'),
            i
        )
        
        if not ptr:
            raise RuntimeError("get_y returned NULL")
        
        # Extract results
        y = ptr[0].decode('utf-8') if ptr[0] else "0"
        k0_prev = ptr[1].decode('utf-8') if ptr[1] else "0"
        
        # Free the array
        self.lib.free_string_array(ptr, 2)
        
        return y, k0_prev
    
    def get_p(self, x0: str, x1: str, D: str, A: str) -> str:
        """Call get_p function directly (for precision testing)."""
        ptr = self.lib.get_p(
            x0.encode('utf-8'),
            x1.encode('utf-8'),
            D.encode('utf-8'),
            A.encode('utf-8')
        )
        return self._get_string_and_free(ptr)
    
    def wad_exp(self, x: str) -> str:
        """Call wad_exp function directly (for precision testing)."""
        ptr = self.lib.wad_exp(x.encode('utf-8'))
        if not ptr:
            return None  # Indicates overflow
        return self._get_string_and_free(ptr)
    
    def __del__(self):
        """Cleanup when done."""
        if hasattr(self, 'lib') and self.lib:
            self.lib.stableswap_math_cleanup()