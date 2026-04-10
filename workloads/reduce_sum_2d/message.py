from typing import Any, Dict


def variant_message(ctx: Dict[str, Any], variant: Dict[str, Any], settings: Dict[str, Any]) -> str:
    """
    Generate a message showing GPU memory bandwidth for reduce_sum_2d.
    
    Args:
        ctx: Context containing config and test_config
        variant: Dictionary containing variant data including timing info
        settings: Test settings from data.py
        
    Returns:
        Formatted string with memory bandwidth information or just variant name
    """
    # Extract problem size - adjust based on workload
    n = settings.get("n", 0)
    if n <= 0:
        return str(variant.get("variant", "unknown"))
    
    # Get timing information from variant row
    custom_ms = variant.get("custom_ms")
    if custom_ms is None or custom_ms <= 0:
        # No timing info available, just return variant name
        return str(variant.get("variant", "unknown"))
    
    # Calculate memory bandwidth - adjust calculation based on workload
    bytes_per_element = 4  # float32, adjust if needed
    total_bytes = n * bytes_per_element  # basic read, adjust based on actual access pattern
    
    # Convert to GB and time to seconds
    total_gb = total_bytes / (1024**3)
    time_s = custom_ms / 1000.0
    
    if time_s > 0:
        bandwidth_gb_s = total_gb / time_s
        variant_name = str(variant.get("variant", "unknown"))
        return f"{variant_name} {bandwidth_gb_s:.2f} GB/s"
    else:
        return str(variant.get("variant", "unknown"))
