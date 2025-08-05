"""
CMake build automation for C++ library.
Can be run directly: uv run benchmark/builder.py
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional
import shutil

# Add parent directory to path for imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import click

console = Console()


class CppBuilder:
    """Handles building the C++ library with CMake."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.build_dir = self.project_root / "build"
        self.lib_path = None
        
    def check_dependencies(self) -> bool:
        """Check if required build tools are available."""
        required = ["cmake", "make", "g++"]
        missing = []
        
        for tool in required:
            if not shutil.which(tool):
                missing.append(tool)
        
        if missing:
            console.print(f"[red]Missing required tools: {', '.join(missing)}[/red]")
            return False
        
        return True
    
    def clean(self):
        """Clean the build directory."""
        if self.build_dir.exists():
            console.print(f"[yellow]Cleaning build directory...[/yellow]")
            shutil.rmtree(self.build_dir)
    
    def configure(self, clean_build: bool = False) -> bool:
        """Configure the CMake project."""
        if clean_build:
            self.clean()
        
        self.build_dir.mkdir(exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Configuring CMake...", total=None)
            
            result = subprocess.run(
                ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
                cwd=self.build_dir,
                capture_output=True,
                text=True
            )
            
            progress.update(task, completed=True)
        
        if result.returncode != 0:
            console.print("[red]CMake configuration failed:[/red]")
            console.print(result.stderr)
            return False
        
        console.print("[green]✓ CMake configured successfully[/green]")
        return True
    
    def build(self, jobs: int = 4) -> bool:
        """Build the C++ library."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Building C++ library...", total=None)
            
            result = subprocess.run(
                ["make", f"-j{jobs}", "stableswap_math_simple"],
                cwd=self.build_dir,
                capture_output=True,
                text=True
            )
            
            progress.update(task, completed=True)
        
        if result.returncode != 0:
            console.print("[red]Build failed:[/red]")
            console.print(result.stderr)
            return False
        
        # Find the built library
        lib_patterns = ["libstableswap_math_simple.so", "libstableswap_math_simple.dylib"]
        for pattern in lib_patterns:
            lib_path = self.build_dir / "lib" / pattern
            if lib_path.exists():
                self.lib_path = lib_path
                break
        
        if not self.lib_path:
            console.print("[red]Built library not found[/red]")
            return False
        
        console.print(f"[green]✓ Library built: {self.lib_path}[/green]")
        return True
    
    def full_build(self, clean: bool = False, jobs: int = 4) -> Optional[Path]:
        """Perform a full build of the C++ library."""
        console.print("[bold blue]Building C++ StableswapMath library[/bold blue]")
        
        if not self.check_dependencies():
            return None
        
        if not self.configure(clean_build=clean):
            return None
        
        if not self.build(jobs=jobs):
            return None
        
        return self.lib_path


@click.command()
@click.option('--clean', is_flag=True, help='Clean build before compiling')
@click.option('--jobs', '-j', default=4, help='Number of parallel build jobs')
def main(clean, jobs):
    """Build the C++ StableswapMath library."""
    builder = CppBuilder()
    lib_path = builder.full_build(clean=clean, jobs=jobs)
    
    if lib_path:
        console.print(f"\n[green]✅ Build successful![/green]")
        console.print(f"[blue]Library location: {lib_path}[/blue]")
        return 0
    else:
        console.print("[red]❌ Build failed![/red]")
        return 1


if __name__ == '__main__':
    sys.exit(main())