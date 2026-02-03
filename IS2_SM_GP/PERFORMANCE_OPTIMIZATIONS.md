# Performance Optimization Guide

Based on timing analysis, here are optimization strategies to speed up the GPSat training pipeline.

## Current Performance Breakdown (from timing_log.txt)

- **Total time: 88.72 min (1.48 hr)**
- **Run LocalExpertOI: 65.03 min (73%)** ⚠️ **MAIN BOTTLENECK**
- **Load data around target date: 16.21 min (18%)**
- **Process data: 0.72 min (1%)**
- **Run LocalExpertOI with smoothed hyperparameters: 6.16 min (7%)**

## Optimization Strategies

### 1. GP Training Optimization (65 min → target: 30-40 min)

**Current settings:**
```python
'max_iter': 10000,
'expert_spacing': 400_000,
'training_radius': 500_000,
'model_type': 'GPflowSGPRModel',
'N_subsample': 1,
```

**Recommended changes:**

#### A. Reduce Maximum Iterations
```python
'max_iter': 5000,  # Reduce from 10000 (50% reduction)
# or even more aggressive:
'max_iter': 3000,  # 70% reduction
```
**Expected impact:** 20-30% faster training (13-20 min saved)
**Trade-off:** May need to check convergence, but early stopping should handle this

#### B. Increase Expert Spacing (Fewer Experts)
```python
'expert_spacing': 600_000,  # Increase from 400_000 (50% fewer experts)
# or even more:
'expert_spacing': 800_000,  # 75% fewer experts
```
**Expected impact:** 30-50% faster (20-30 min saved)
**Trade-off:** Lower spatial resolution of hyperparameters

#### C. Reduce Training Radius
```python
'training_radius': 400_000,  # Reduce from 500_000 (20% less data per expert)
# or more aggressive:
'training_radius': 300_000,  # 40% less data per expert
```
**Expected impact:** 15-25% faster (10-16 min saved)
**Trade-off:** Less data per expert, potentially less accurate

#### D. Increase Data Subsampling
```python
'N_subsample': 2,  # Skip every other point (50% less data)
# or more:
'N_subsample': 3,  # 67% less data
```
**Expected impact:** 20-30% faster (13-20 min saved)
**Trade-off:** Less data for training, potentially less accurate

#### E. Use Faster Model Type
```python
'model_type': 'GPflowVFFModel',  # Variational Fourier Features (faster)
# or:
'model_type': 'GPflowASVGPModel',  # Another fast variant
```
**Expected impact:** 10-20% faster (6-13 min saved)
**Trade-off:** May have different accuracy characteristics

### 2. Data Loading Optimization (16.21 min → target: 5-10 min)

**Already implemented:**
- ✅ SMAP file caching (first run downloads, subsequent runs use cache)

**Additional optimizations:**

#### A. Parallel Day Loading
Load multiple days in parallel using multiprocessing:
```python
from multiprocessing import Pool

def load_day_parallel(args):
    date_str, IS2, config = args
    return load_smap_data_for_date(date_str, IS2, config)

# In load_data_around_target_date:
with Pool(processes=4) as pool:
    dates = [current_date_str for current_dt in date_range]
    results = pool.map(load_day_parallel, [(d, IS2, config) for d in dates])
```
**Expected impact:** 50-70% faster (8-11 min saved)
**Trade-off:** Higher memory usage

#### B. Use Prediction Day Only for SMAP
```python
'smap_use_prediction_day_only': True,  # Only load SMAP for target date
```
**Expected impact:** 80-90% faster SMAP loading (13-14 min saved)
**Trade-off:** Less SMAP data, but may be acceptable if along-track data is primary

### 3. Combined Optimization Strategy

**Conservative (target: 50-60 min total):**
```python
'max_iter': 5000,           # 50% reduction
'expert_spacing': 500_000,  # 25% fewer experts
'training_radius': 400_000, # 20% less data
'N_subsample': 2,           # 50% less data
'smap_use_prediction_day_only': True,  # Faster SMAP loading
```
**Expected:** ~50-60 min total (30-40 min saved)

**Aggressive (target: 30-40 min total):**
```python
'max_iter': 3000,           # 70% reduction
'expert_spacing': 800_000,  # 75% fewer experts
'training_radius': 300_000, # 40% less data
'N_subsample': 3,           # 67% less data
'model_type': 'GPflowVFFModel',  # Faster model
'smap_use_prediction_day_only': True,
```
**Expected:** ~30-40 min total (50-60 min saved)

### 4. GPU Optimization

**Check GPU usage:**
```python
# Add to config or check manually
import tensorflow as tf
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("GPU memory growth:", tf.config.experimental.get_memory_growth(...))
```

**Ensure GPU is being used:**
- Current code sets `CUDA_VISIBLE_DEVICES = '-1'` (CPU only!)
- Remove or comment out this line to use GPU
- GPU can be 10-50x faster for GP training

### 5. Memory Optimization

**Current peak memory: 5562 MB (5.5 GB)**

To reduce memory:
- Increase `N_subsample` (already mentioned)
- Reduce `training_radius`
- Process fewer days at once
- Use batch processing for large datasets

## Recommended Quick Wins

1. **Enable GPU** (if available): Remove `CUDA_VISIBLE_DEVICES = '-1'`
2. **Reduce max_iter to 5000**: Minimal accuracy loss, ~20% faster
3. **Use prediction day only for SMAP**: ~14 min saved
4. **Increase N_subsample to 2**: ~13 min saved

**Combined quick wins: ~40-50 min saved (from 88 min to ~40-50 min)**

## Monitoring Performance

After making changes, check:
- `timing_log.txt` for updated timings
- Prediction accuracy (may need to verify quality hasn't degraded)
- Memory usage (should stay reasonable)

## Testing Strategy

1. Start with conservative optimizations
2. Run on a test date
3. Compare results quality
4. If acceptable, try more aggressive optimizations
5. Iterate until finding best speed/accuracy trade-off



