"""
Progress bar for smina molecular docking
Monitors smina output and shows progress based on completed dockings
"""

import subprocess
import re
import time
import sys
from tqdm import tqdm
import threading
import queue
import os

class SminaProgressMonitor:
    def __init__(self, sdf_file, total_molecules=None):
        self.sdf_file = sdf_file
        self.total_molecules = total_molecules or self.count_molecules_in_sdf(sdf_file)
        self.completed = 0
        self.progress_bar = None
        
    def count_molecules_in_sdf(self, sdf_file):
        """Count total molecules in SDF file by counting $$$$  delimiters"""
        try:
            with open(sdf_file, 'r') as f:
                content = f.read()
                # Count molecule separators in SDF format
                return content.count('$$$$')
        except FileNotFoundError:
            print(f"Error: SDF file {sdf_file} not found")
            return 0
    
    def run_smina_with_progress(self, smina_cmd, output_file=None):
        """
        Run smina command and monitor progress
        
        Args:
            smina_cmd: List of smina command arguments
            output_file: Optional output file to monitor for completed dockings
        """
        print(f"Starting docking of {self.total_molecules} molecules...")
        
        # Initialize progress bar
        self.progress_bar = tqdm(total=self.total_molecules, 
                                desc="Docking Progress", 
                                unit="molecules")
        
        # Start smina process
        process = subprocess.Popen(
            smina_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor output in separate thread
        output_queue = queue.Queue()
        
        def monitor_output(proc, q):
            """Monitor smina stdout for progress indicators"""
            for line in proc.stdout:
                q.put(('stdout', line))
            proc.stdout.close()
        
        def monitor_stderr(proc, q):
            """Monitor smina stderr for progress indicators"""
            for line in proc.stderr:
                q.put(('stderr', line))
            proc.stderr.close()
        
        # Start monitoring threads
        stdout_thread = threading.Thread(target=monitor_output, args=(process, output_queue))
        stderr_thread = threading.Thread(target=monitor_stderr, args=(process, output_queue))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Monitor progress
        if output_file:
            # Also monitor output file growth
            file_monitor_thread = threading.Thread(
                target=self.monitor_output_file, 
                args=(output_file, output_queue)
            )
            file_monitor_thread.daemon = True
            file_monitor_thread.start()
        
        # Process output and update progress
        while process.poll() is None or not output_queue.empty():
            try:
                stream, line = output_queue.get(timeout=0.1)
                self.parse_progress_from_output(line)
            except queue.Empty:
                continue
        
        # Wait for process to complete
        process.wait()
        
        # Final update
        if self.completed < self.total_molecules:
            self.progress_bar.update(self.total_molecules - self.completed)
        
        self.progress_bar.close()
        
        if process.returncode == 0:
            print(f"\nDocking completed successfully! {self.total_molecules} molecules processed.")
        else:
            print(f"\nSmina exited with error code: {process.returncode}")
        
        return process.returncode
    
    def parse_progress_from_output(self, line):
        """Parse smina output line for progress indicators"""
        # Look for patterns that indicate a molecule has been processed
        
        patterns = [
            r'Refining results',  # Common smina output when finishing a molecule
            r'mode\s+\|\s+affinity',  # Results table header
            r'Writing output',  # Output writing phase
            r'.*\.pdbqt',  # PDBQT file mentions
        ]
        
        for pattern in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                self.update_progress()
                break
    
    def monitor_output_file(self, output_file, output_queue):
        """Monitor output file size/content for progress"""
        last_size = 0
        molecules_found = 0
        
        while True:
            try:
                if os.path.exists(output_file):
                    current_size = os.path.getsize(output_file)
                    if current_size > last_size:
                        # Count molecules in output file
                        with open(output_file, 'r') as f:
                            content = f.read()
                            new_molecules = content.count('$$$$')
                            
                            if new_molecules > molecules_found:
                                # Update progress based on output file
                                diff = new_molecules - molecules_found
                                for _ in range(diff):
                                    output_queue.put(('file', 'molecule_completed'))
                                molecules_found = new_molecules
                        
                        last_size = current_size
                
                time.sleep(1)  # Check every second
                
            except (FileNotFoundError, OSError):
                time.sleep(1)
                continue
    
    def update_progress(self):
        """Update progress bar"""
        if self.completed < self.total_molecules:
            self.completed += 1
            self.progress_bar.update(1)
            self.progress_bar.set_postfix({
                'Completed': f'{self.completed}/{self.total_molecules}'
            })


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run smina with progress bar')
    parser.add_argument('--ligand', '-l', required=True, help='Input SDF file')
    parser.add_argument('--receptor', '-r', required=True, help='Receptor file')
    parser.add_argument('--output', '-o', help='Output file')
    
    # Search space definition (mutually exclusive groups)
    search_group = parser.add_mutually_exclusive_group()
    
    # Manual search box
    parser.add_argument('--center_x', type=float, help='Search center X coordinate')
    parser.add_argument('--center_y', type=float, help='Search center Y coordinate')
    parser.add_argument('--center_z', type=float, help='Search center Z coordinate')
    parser.add_argument('--size_x', type=float, default=20, help='Search size X')
    parser.add_argument('--size_y', type=float, default=20, help='Search size Y')
    parser.add_argument('--size_z', type=float, default=20, help='Search size Z')
    
    # Autobox options
    parser.add_argument('--autobox_ligand', help='Reference ligand file for autobox')
    parser.add_argument('--autobox_add', type=int, help='Add buffer to autobox (Angstroms)')
    
    # Other smina parameters
    parser.add_argument('--exhaustiveness', type=int, default=16, help='Exhaustiveness')
    parser.add_argument('--num_modes', type=int, default=1, help='Number of modes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducible results')
    parser.add_argument('--energy_range', type=float, help='Energy range for output poses')
    parser.add_argument('--cpu', type=int, help='Number of CPUs to use')
    
    args = parser.parse_args()
    
    # Validate search space options
    manual_center = all([args.center_x is not None, args.center_y is not None, args.center_z is not None])
    has_autobox = args.autobox_ligand is not None
    
    if not manual_center and not has_autobox:
        parser.error("Must specify either manual search box (--center_x, --center_y, --center_z) or autobox (--autobox_ligand)")
    
    if manual_center and has_autobox:
        parser.error("Cannot specify both manual search box and autobox options")
    
    # Build smina command
    smina_cmd = [
        'smina',
        '--receptor', args.receptor,
        '--ligand', args.ligand,
        '--exhaustiveness', str(args.exhaustiveness),
        '--num_modes', str(args.num_modes)
    ]
    
    # Add output file
    if args.output:
        smina_cmd.extend(['--out', args.output])
    
    # Add search space definition
    if manual_center:
        smina_cmd.extend([
            '--center_x', str(args.center_x),
            '--center_y', str(args.center_y),
            '--center_z', str(args.center_z),
            '--size_x', str(args.size_x),
            '--size_y', str(args.size_y),
            '--size_z', str(args.size_z)
        ])
    elif has_autobox:
        smina_cmd.extend(['--autobox_ligand', args.autobox_ligand])
        if args.autobox_add is not None:
            smina_cmd.extend(['--autobox_add', str(args.autobox_add)])
    
    # Add optional parameters
    if args.seed is not None:
        smina_cmd.extend(['--seed', str(args.seed)])
 
    if args.energy_range is not None:
        smina_cmd.extend(['--energy_range', str(args.energy_range)]) 
    
    if args.cpu is not None:
        smina_cmd.extend(['--cpu', str(args.cpu)])
    
    # Create progress monitor
    monitor = SminaProgressMonitor(args.ligand)
    
    # Print command for verification
    print("Running smina command:")
    print(" ".join(smina_cmd))
    print()
    
    # Run with progress tracking
    return_code = monitor.run_smina_with_progress(smina_cmd, args.output)
    
    sys.exit(return_code)


if __name__ == "__main__":
    main()

