# rTAS Tests

Minimal, faithful tests for TAS and rTAS that use the actual API in this repository.

## Test Files

### `test_tas_simple.py`
Tests core TAS properties using functions from `scripts/tas_figures.py`:
- **Strip-sine area law** (§6.1): Verifies Δh = 2α·Area
- **Cost-memory trade-off** (Theorem 1): Prescribed dynamics cost more energy than geometric lift
- **Small-loop quadratic law** (Proposition 3): E_excess ∝ (Δh)² for small loops
- **Zero holonomy**: Geometric lift produces no holonomy

### `test_rtas_simple.py`
Tests rTAS properties using functions from `scripts/rtas_figures.py`:
- **Projection constraint** (Definition 7): Reflective lift satisfies c = Φ_m(u,v)
- **Lambda effect**: Higher λ reduces model adaptation

## Running Tests

```bash
# Run TAS tests
python tests/test_tas_simple.py

# Run rTAS tests  
python tests/test_rtas_simple.py
```

No additional dependencies required beyond the main requirements.


## Test Results

All tests pass with the current implementation:
- Strip-sine area law: ✓ (error < 1%)
- Energy comparison: ✓ (prescribed > geometric)
- Small-loop scaling: ✓ (exponent ≈ 2.0)
- Projection consistency: ✓ (error < 1e-16)
- Lambda ordering: ✓ (higher λ → less model change)
