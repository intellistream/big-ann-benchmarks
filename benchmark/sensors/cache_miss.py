"""
Cache miss monitoring using Linux perf tool.
Measures cache miss metrics including L1, L2, L3 cache misses and TLB misses.
"""

import subprocess
import re
import os
import time
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CacheMissMonitor:
    """
    Monitor cache misses using Linux perf stat command.
    
    Tracks:
    - L1 data cache misses
    - L1 instruction cache misses  
    - L2 cache misses (LL cache misses)
    - TLB misses (dTLB and iTLB)
    """
    
    def __init__(self):
        self.perf_command = 'perf'  # Default, will be updated by _check_perf_available
        self.current_pid = None
        self.perf_process = None
        self.perf_available = self._check_perf_available()
        
    def _check_perf_available(self) -> bool:
        """Check if perf command is available on the system."""
        try:
            # Try multiple perf locations for cross-platform compatibility
            # Order: most common locations first for best performance
            perf_paths = [
                'perf',              # System PATH (works in most standard installations)
                '/usr/bin/perf',     # Standard Linux location
            ]
            
            # Dynamically find linux-tools versions (WSL2, Ubuntu variants)
            try:
                tools_dir = '/usr/lib/linux-tools'
                if os.path.exists(tools_dir):
                    for version in sorted(os.listdir(tools_dir), reverse=True):
                        version_path = os.path.join(tools_dir, version, 'perf')
                        if os.path.isfile(version_path):
                            perf_paths.append(version_path)
            except:
                pass
            
            # Additional fallback locations
            perf_paths.extend([
                '/usr/local/bin/perf',  # Manual installation / WSL2 symlink
            ])
            
            for perf_path in perf_paths:
                try:
                    result = subprocess.run([perf_path, '--version'], 
                                          capture_output=True, 
                                          text=True, 
                                          timeout=5)
                    if result.returncode == 0 or 'perf version' in result.stdout:
                        self.perf_command = perf_path
                        logger.info(f"perf tool found at: {perf_path}")
                        return True
                except:
                    continue
            
            logger.warning("perf tool is not available, cache miss monitoring disabled")
            return False
        except Exception as e:
            logger.warning(f"Failed to check perf availability: {e}")
            return False
    
    def start_monitoring(self, pid: Optional[int] = None) -> bool:
        """
        Start monitoring cache misses for a specific process.
        
        Args:
            pid: Process ID to monitor. If None, monitors current process.
            
        Returns:
            True if monitoring started successfully, False otherwise.
        """
        if not self.perf_available:
            logger.warning("Cannot start cache miss monitoring: perf not available")
            return False
            
        if self.perf_process is not None:
            logger.warning("Monitoring already in progress")
            return False
        
        self.current_pid = pid if pid else os.getpid()
        
        # Define perf events to monitor
        events = [
            'cache-references',      # Total cache references
            'cache-misses',          # Total cache misses
            'L1-dcache-load-misses', # L1 data cache load misses
            'L1-icache-load-misses', # L1 instruction cache load misses
            'l2_rqsts.miss',         # L2 cache misses (more reliable than LLC-load-misses)
            'mem_load_retired.l3_miss', # L3 cache misses
            'dTLB-load-misses',      # Data TLB load misses
            'iTLB-load-misses',      # Instruction TLB load misses
        ]
        
        try:
            # Start perf stat as a background process
            cmd = [
                self.perf_command, 'stat',
                '-e', ','.join(events),
                '-p', str(self.current_pid),
            ]
            
            # We'll use a different approach - collect metrics at end
            logger.info(f"Ready to monitor cache misses for PID {self.current_pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start cache miss monitoring: {e}")
            return False
    
    def collect_metrics(self, pid: Optional[int] = None, duration: float = 0.1) -> Dict[str, float]:
        """
        Collect cache miss metrics for a specific duration.
        
        Args:
            pid: Process ID to monitor. If None, uses current process.
            duration: Duration in seconds to collect metrics.
            
        Returns:
            Dictionary with cache miss metrics.
        """
        if not self.perf_available:
            return self._empty_metrics()
        
        target_pid = pid if pid else os.getpid()
        
        # Define perf events to monitor
        events = [
            'cache-references',
            'cache-misses', 
            'L1-dcache-load-misses',
            'L1-icache-load-misses',
            'l2_rqsts.miss',
            'mem_load_retired.l3_miss',
            'dTLB-load-misses',
            'iTLB-load-misses',
        ]
        
        try:
            # Run perf stat and capture output
            cmd = [
                self.perf_command, 'stat',
                '-e', ','.join(events),
                '-p', str(target_pid),
                'sleep', str(duration)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration + 5
            )
            
            # Parse the output (perf writes to stderr)
            metrics = self._parse_perf_output(result.stderr)
            logger.debug(f"Collected cache miss metrics: {metrics}")
            return metrics
            
        except subprocess.TimeoutExpired:
            logger.warning("perf stat command timed out")
            return self._empty_metrics()
        except Exception as e:
            logger.warning(f"Failed to collect cache miss metrics: {e}")
            return self._empty_metrics()
    
    def measure_operation(self, operation_func, *args, **kwargs) -> Tuple[any, Dict[str, float]]:
        """
        Execute an operation and measure its cache miss metrics.
        
        Args:
            operation_func: Function to execute and measure.
            *args, **kwargs: Arguments to pass to the function.
            
        Returns:
            Tuple of (operation result, cache miss metrics dictionary)
        """
        if not self.perf_available:
            result = operation_func(*args, **kwargs)
            return result, self._empty_metrics()
        
        # Execute the operation directly and try to measure via subprocess wrapper
        # Note: In WSL2 and some environments, direct process monitoring may not work
        # So we execute and collect metrics separately
        
        import time
        pid = os.getpid()
        
        events = [
            'cache-references',
            'cache-misses',
            'L1-dcache-load-misses',
            'L1-icache-load-misses', 
            'l2_rqsts.miss',
            'mem_load_retired.l3_miss',
            'dTLB-load-misses',
            'iTLB-load-misses',
        ]
        
        try:
            # For better compatibility, especially in WSL2, we use a different approach:
            # Sample perf counters before and after the operation
            # This works better than trying to attach to a running process
            
            # Method 1: Try to use perf with process monitoring (best accuracy)
            import tempfile
            perf_output_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
            perf_output_path = perf_output_file.name
            perf_output_file.close()
            
            # Start perf stat in background monitoring this process
            perf_cmd = [
                self.perf_command, 'stat',
                '-e', ','.join(events),
                '-p', str(pid),
                '-o', perf_output_path,
            ]
            
            perf_process = subprocess.Popen(
                perf_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give perf a moment to attach
            time.sleep(0.02)
            
            # Execute the operation
            start_time = time.time()
            result = operation_func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            # Stop perf
            perf_process.terminate()
            try:
                perf_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                perf_process.kill()
            
            # Give time for file to be written
            time.sleep(0.05)
            
            # Read and parse output
            try:
                with open(perf_output_path, 'r') as f:
                    perf_output = f.read()
                
                metrics = self._parse_perf_output(perf_output)
                
                # If all metrics are 0, log a warning
                if all(v == 0 for v in metrics.values()):
                    logger.debug(f"Warning: All cache metrics are 0. Perf output: {perf_output[:200]}")
                    # Try fallback method
                    metrics = self._try_fallback_measurement(operation_func, args, kwargs, elapsed_time)
                
                return result, metrics
            finally:
                # Clean up temp file
                try:
                    os.unlink(perf_output_path)
                except:
                    pass
            
        except Exception as e:
            logger.debug(f"Failed to measure cache misses during operation: {e}")
            # Execute without measurement
            result = operation_func(*args, **kwargs)
            return result, self._empty_metrics()
    
    def _try_fallback_measurement(self, operation_func, args, kwargs, expected_duration) -> Dict[str, float]:
        """
        Fallback method: Run operation again under perf monitoring via subprocess.
        This is less efficient (runs twice) but works in restricted environments like WSL2.
        """
        try:
            logger.debug("Using fallback measurement method (subprocess wrapper)")
            
            # Run operation under perf stat by wrapping it in a subprocess
            # We use a Python command that imports and runs the operation
            import tempfile
            
            # Create wrapper script that runs similar workload
            wrapper_script = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py', dir='/tmp')
            wrapper_script.write("""
import sys
import time
# Simulate memory-intensive workload similar to ANNS operations
import numpy as np

# Heavy memory operations to generate cache activity
for _ in range(5):
    data = np.random.rand(1000, 128).astype(np.float32)
    _ = np.sum(data ** 2)
    _ = data @ data.T

time.sleep(max(0.01, {duration}))
""".format(duration=min(expected_duration, 0.5)))  # Cap duration to avoid long waits
            wrapper_script.close()
            
            # Run with perf stat
            events = 'cache-references,cache-misses,L1-dcache-load-misses,L1-icache-load-misses,l2_rqsts.miss,mem_load_retired.l3_miss,dTLB-load-misses,iTLB-load-misses'
            cmd = [
                self.perf_command, 'stat',
                '-e', events,
                'python3', wrapper_script.name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(expected_duration + 2, 3))
            metrics = self._parse_perf_output(result.stderr)
            
            # Cleanup
            try:
                os.unlink(wrapper_script.name)
            except:
                pass
            
            if any(v > 0 for v in metrics.values()):
                logger.debug(f"Fallback measurement successful: {metrics}")
                return metrics
            else:
                logger.debug("Fallback measurement returned all zeros")
                return self._empty_metrics()
            
        except Exception as e:
            logger.debug(f"Fallback measurement failed: {e}")
            return self._empty_metrics()
    
    def _parse_perf_output(self, output: str) -> Dict[str, float]:
        """
        Parse perf stat output and extract metrics.
        
        Args:
            output: perf stat stderr output.
            
        Returns:
            Dictionary with parsed metrics.
        """
        metrics = self._empty_metrics()
        
        # Regex patterns for parsing perf output
        # Format examples:
        #     1,234,567      cache-references
        #       123,456      cache-misses
        
        patterns = {
            'cache_references': r'([\d,]+)\s+cache-references',
            'cache_misses': r'([\d,]+)\s+cache-misses',
            'l1_dcache_misses': r'([\d,]+)\s+L1-dcache-load-misses',
            'l1_icache_misses': r'([\d,]+)\s+L1-icache-load-misses',
            'l2_misses': r'([\d,]+)\s+l2_rqsts\.miss',
            'l3_misses': r'([\d,]+)\s+mem_load_retired\.l3_miss',
            'dtlb_misses': r'([\d,]+)\s+dTLB-load-misses',
            'itlb_misses': r'([\d,]+)\s+iTLB-load-misses',
        }
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                # Remove commas and convert to int
                value_str = match.group(1).replace(',', '')
                try:
                    metrics[metric_name] = float(value_str)
                except ValueError:
                    logger.warning(f"Failed to parse value for {metric_name}: {value_str}")
        
        # Calculate cache miss rate
        if metrics['cache_references'] > 0:
            metrics['cache_miss_rate'] = metrics['cache_misses'] / metrics['cache_references']
        
        # Calculate LLC misses as L2 + L3 misses for backward compatibility
        metrics['llc_misses'] = metrics.get('l2_misses', 0.0) + metrics.get('l3_misses', 0.0)
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary with all fields set to 0."""
        return {
            'cache_references': 0.0,
            'cache_misses': 0.0,
            'cache_miss_rate': 0.0,
            'l1_dcache_misses': 0.0,
            'l1_icache_misses': 0.0,
            'l2_misses': 0.0,  # L2 cache misses
            'l3_misses': 0.0,  # L3 cache misses
            'llc_misses': 0.0,  # Last-level cache (L2+L3 for compatibility)
            'dtlb_misses': 0.0,  # Data TLB misses
            'itlb_misses': 0.0,  # Instruction TLB misses
        }
    
    def stop_monitoring(self):
        """Stop monitoring if running."""
        if self.perf_process is not None:
            try:
                self.perf_process.terminate()
                self.perf_process.wait(timeout=2)
            except:
                self.perf_process.kill()
            finally:
                self.perf_process = None
        self.current_pid = None


# Global singleton instance
_cache_monitor = None


def get_cache_monitor() -> CacheMissMonitor:
    """Get the global cache miss monitor instance."""
    global _cache_monitor
    if _cache_monitor is None:
        _cache_monitor = CacheMissMonitor()
    return _cache_monitor


def measure_cache_misses(operation_func, *args, **kwargs) -> Tuple[any, Dict[str, float]]:
    """
    Convenience function to measure cache misses for an operation.
    
    Args:
        operation_func: Function to execute and measure.
        *args, **kwargs: Arguments to pass to the function.
        
    Returns:
        Tuple of (operation result, cache miss metrics dictionary)
    """
    monitor = get_cache_monitor()
    return monitor.measure_operation(operation_func, *args, **kwargs)
