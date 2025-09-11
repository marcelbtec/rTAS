"""
Minimal, faithful tests for TAS that use the actual API in tas_figures.py
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.tas_figures import GeometricTASExplorer


def circle_path(R=1.0, N=4000):
    """Generate a circular path in cognitive space."""
    t = np.linspace(0, 2*np.pi, N, endpoint=True)
    c = np.column_stack([R*np.cos(t), R*np.sin(t)])
    dt = t[1] - t[0]
    return c, dt


def energy_with_hidden(uv, h, dt):
    """Compute total energy including hidden fiber contribution."""
    v_uv = np.diff(uv, axis=0) / dt           # (N-1,2)
    v_h  = np.diff(h) / dt                    # (N-1,)
    return float(np.sum((v_uv**2).sum(axis=1) + v_h**2) * dt)


def test_strip_sine_area_law():
    """§6.1: Verify Δh = 2α · Area for prescribed strip-sine dynamics."""
    expl = GeometricTASExplorer()
    alpha, kappa, R = 0.5, 0.3, 0.8
    c, dt = circle_path(R)
    
    uv, h = expl.compute_strip_sine_prescribed_dynamics(c, kappa=kappa, alpha=alpha)
    
    # Expected holonomy: 2 * alpha * area
    expected = 2 * alpha * np.pi * R**2
    actual = h[-1] - h[0]
    
    print(f"Strip-sine area law: expected={expected:.4f}, actual={actual:.4f}")
    assert np.isclose(actual, expected, rtol=1e-2), f"Expected {expected}, got {actual}"


def test_geometric_vs_prescribed_energy():
    """Theorem 1: Prescribed vertical motion costs extra energy vs. geometric baseline."""
    expl = GeometricTASExplorer()
    alpha, kappa, R = 0.4, 0.5, 0.7
    c, dt = circle_path(R)

    # Geometric lift (no holonomy)
    uv_g, h_g = expl.compute_strip_sine_geometric_lift(c, kappa=kappa)
    # Prescribed dynamics (with holonomy)
    uv_p, h_p = expl.compute_strip_sine_prescribed_dynamics(c, kappa=kappa, alpha=alpha)

    # Compute energies including hidden channel
    E_geom = energy_with_hidden(uv_g, np.zeros_like(h_p), dt)   # no hidden motion
    E_pres = energy_with_hidden(uv_p, h_p, dt)
    
    print(f"Energy comparison: E_geom={E_geom:.4f}, E_pres={E_pres:.4f}, excess={E_pres-E_geom:.4f}")
    assert E_pres >= E_geom - 1e-10, "Prescribed dynamics should have higher energy"


def test_small_loop_quadratic_law():
    """Proposition 3: E_excess ∝ (Δh)² for small loops."""
    expl = GeometricTASExplorer()
    alpha, kappa = 0.5, 0.3
    radii = np.array([0.10, 0.15, 0.20, 0.25, 0.30])  # Small loops

    holonomies = []
    excess_energies = []
    
    for R in radii:
        c, dt = circle_path(R, N=2000)  # Dense sampling
        
        # Geometric baseline
        uv_g, h_g = expl.compute_strip_sine_geometric_lift(c, kappa=kappa)
        E_geom = energy_with_hidden(uv_g, np.zeros_like(h_g), dt)
        
        # Prescribed dynamics
        uv_p, h_p = expl.compute_strip_sine_prescribed_dynamics(c, kappa=kappa, alpha=alpha)
        E_pres = energy_with_hidden(uv_p, h_p, dt)
        
        holonomies.append(h_p[-1] - h_p[0])
        excess_energies.append(E_pres - E_geom)

    # Fit power law: E_excess = a * h^b
    H = np.array(holonomies)
    E = np.array(excess_energies)
    
    # Log-log regression
    log_H = np.log(H[H > 0])
    log_E = np.log(E[H > 0])
    exponent = np.polyfit(log_H, log_E, 1)[0]
    
    print(f"Small-loop scaling exponent: {exponent:.3f} (expected: 2.0)")
    assert 1.8 < exponent < 2.2, f"Expected quadratic scaling (2.0), got {exponent}"


def test_geometric_lift_zero_holonomy():
    """Geometric lift should produce zero holonomy."""
    expl = GeometricTASExplorer()
    kappa, R = 0.5, 1.0
    c, dt = circle_path(R)
    
    uv, h = expl.compute_strip_sine_geometric_lift(c, kappa=kappa)
    
    holonomy = h[-1] - h[0]
    print(f"Geometric lift holonomy: {holonomy:.2e} (should be ~0)")
    assert abs(holonomy) < 1e-10, "Geometric lift should have zero holonomy"


if __name__ == "__main__":
    print("Running minimal TAS tests...\n")
    
    test_strip_sine_area_law()
    print("✓ Strip-sine area law test passed\n")
    
    test_geometric_vs_prescribed_energy()
    print("✓ Cost-memory trade-off test passed\n")
    
    test_small_loop_quadratic_law()
    print("✓ Small-loop quadratic law test passed\n")
    
    test_geometric_lift_zero_holonomy()
    print("✓ Geometric lift zero holonomy test passed\n")
    
    print("All tests passed!")
