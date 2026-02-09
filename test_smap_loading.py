#!/usr/bin/env python3
"""
Test script for SMAP data loading functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the function from the main script
from IS2_SMAP_GPSat_train import load_smap_data_for_date, read_IS2SITMOGR4

def test_smap_loading():
    """Test SMAP data loading for a specific date"""
    
    print("Testing SMAP data loading...")
    
    # Test date
    test_date = '2019-04-15'
    
    # Load IS2 data for reference
    print("Loading IS2 data for reference...")
    try:
        IS2 = read_IS2SITMOGR4(data_type='netcdf-local', version='003', 
                               local_data_path="/explore/nobackup/people/aapetty/IS2thickness/rel006/run_adapt_4/final_data_gridded/").rename({'longitude':'lon','latitude':'lat'})
        print("IS2 data loaded successfully")
    except Exception as e:
        print(f"Error loading IS2 data: {e}")
        return
    
    # Test configuration
    config = {
        'smap_thickness_min': 0.05,
        'smap_thickness_max': 2.0
    }
    
    # Test SMAP data loading
    print(f"Loading SMAP data for {test_date}...")
    try:
        smap_data, smap_data_gridded = load_smap_data_for_date(test_date, IS2, config)
        
        print(f"SMAP data loading test completed:")
        print(f"  - Number of data points: {len(smap_data)}")
        print(f"  - Data columns: {list(smap_data.columns)}")
        print(f"  - Thickness range: {smap_data['ice_thickness'].min():.3f} to {smap_data['ice_thickness'].max():.3f} m")
        print(f"  - Gridded data shape: {smap_data_gridded.dims if hasattr(smap_data_gridded, 'dims') else 'No gridded data'}")
        
        if len(smap_data) > 0:
            print("✓ SMAP data loading test PASSED")
        else:
            print("⚠ SMAP data loading test completed but no data found")
            
    except Exception as e:
        print(f"✗ SMAP data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_smap_loading() 