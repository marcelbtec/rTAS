import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, Arrow
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')


class GeometricTASExplorer:
    """
    Explores the geometric structure of Tangential Action Spaces.
    Focuses on the relationships between C-space trajectories and their P-space projections.
    Implements both diffeomorphisms and fibrations according to the paper's classification.
    """
    
    def __init__(self):
        pass
    def compute_strip_sine_prescribed_dynamics(self, c_path, kappa=0.5, alpha=0.3):
        """
        Computes the physical path using PRESCRIBED DYNAMICS with memory in hidden fiber.
        The system has coordinates (u, v, h) where h is the hidden fiber.
        Projection: Φ(u,v,h) = (c₁,c₂) = (u, v + κ sin u)
        Prescribed dynamics: ḣ = α(c₁ċ₂ - c₂ċ₁) creates memory
        """
        # Initial conditions from cognitive to physical
        c1_0, c2_0 = c_path[0]
        u0 = c1_0
        v0 = c2_0 - kappa * np.sin(u0)  # Inverse of projection
        h0 = 0.0  # Start with zero hidden state
        
        u_path, v_path, h_path = [u0], [v0], [h0]

        for i in range(len(c_path) - 1):
            c1, c2 = c_path[i]
            c1_next, c2_next = c_path[i+1]
            dc1, dc2 = c1_next - c1, c2_next - c2
            u_current = u_path[-1]
            
            # Horizontal lift (geometric part)
            du = dc1
            u_mid = u_current + 0.5*du
            dv = dc2 - kappa * np.cos(u_mid) * dc1
            
            # Vertical component (memory storage)
            # Use midpoint rule for better accuracy: evaluate at midpoint of interval
            c1_mid = (c1 + c1_next) / 2
            c2_mid = (c2 + c2_next) / 2
            dh = alpha * (c1_mid * dc2 - c2_mid * dc1)
            
            u_path.append(u_current + du)
            v_path.append(v_path[-1] + dv)
            h_path.append(h_path[-1] + dh)
            
        return np.column_stack([u_path, v_path]), np.array(h_path)

    def compute_strip_sine_geometric_lift(self, c_path, kappa=0.5):
        """
        Computes the physical path using the unique, energy-minimal GEOMETRIC LIFT.
        This path will have ZERO holonomy in the hidden fiber (ḣ = 0).
        Projection: Φ(u,v,h) = (c₁,c₂) = (u, v + κ sin u)
        """
        # Initial conditions from cognitive to physical
        c1_0, c2_0 = c_path[0]
        u0 = c1_0
        v0 = c2_0 - kappa * np.sin(u0)  # Inverse of projection
        h0 = 0.0
        
        u_path, v_path, h_path = [u0], [v0], [h0]

        for i in range(len(c_path) - 1):
            c1, c2 = c_path[i]
            c1_next, c2_next = c_path[i+1]
            dc1, dc2 = c1_next - c1, c2_next - c2
            u_current = u_path[-1]

            # Horizontal lift only (no vertical component)
            du = dc1
            dv = dc2 - kappa * np.cos(u_current) * dc1
            dh = 0  # Geometric lift has no change in hidden fiber
            
            u_path.append(u_current + du)
            v_path.append(v_path[-1] + dv)
            h_path.append(h_path[-1] + dh)
            
        return np.column_stack([u_path, v_path]), np.array(h_path)

explorer = GeometricTASExplorer()

def energy_squared_speed(path, dt):
    """Calculate energy using squared-speed functional E = ∫||ů||² dt"""
    # path: array (N, d) with coordinates sampled at constant dt
    vel = np.diff(path, axis=0) / dt
    return float(np.sum(np.sum(vel**2, axis=1)) * dt)

def create_strip_sine_detailed(explorer):
    """Create detailed figure for strip-sine system with hidden fiber memory"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1, 1], width_ratios=[1, 1, 1],
                          hspace=0.4, wspace=0.35)
    
    fig.suptitle('Strip-Sine System: Memory in Hidden Fiber', fontsize=16, fontweight='bold')
    
    kappa = 0.5
    alpha = 0.3
    t_fine = np.linspace(0, 2*np.pi, 400, endpoint=True)
    dt = t_fine[1] - t_fine[0]  # Time step for proper integration
    
    # 1. Visible Space Trajectories (u,v)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Show the projection mapping
    u_grid = np.linspace(-2*np.pi, 2*np.pi, 100)
    v_grid = np.linspace(-2.5, 2.5, 100)
    U, V = np.meshgrid(u_grid, v_grid)
    # Show contours of c2 = v + κ sin(u)
    C2 = V + kappa * np.sin(U)
    
    im = ax1.contour(U, V, C2, levels=20, colors='gray', alpha=0.3, linewidths=0.5)
    
    # Plot trajectories for different radii
    for R in [0.8, 1.5]:
        c_path = np.column_stack([R*np.cos(t_fine), R*np.sin(t_fine)])
        p_path, h_path = explorer.compute_strip_sine_prescribed_dynamics(c_path, kappa, alpha)
        ax1.plot(p_path[:, 0], p_path[:, 1], 'k-', linewidth=2)
        ax1.scatter(p_path[0,0], p_path[0,1], color='g', s=80, edgecolor='k', zorder=5)
        ax1.scatter(p_path[-1,0], p_path[-1,1], color='r', marker='s', s=80, edgecolor='k', zorder=5)
        # Note: visible coordinates close perfectly!

    ax1.set_xlabel('$u$', fontsize=12); ax1.set_ylabel('$v$', fontsize=12)
    ax1.set_title('Visible Coordinates Close Perfectly', fontsize=14)
    ax1.grid(True, alpha=0.3); ax1.set_xlim(-2*np.pi, 2*np.pi); ax1.set_ylim(-2.5, 2.5)
    
    # 2. Hidden Fiber Evolution
    ax2 = fig.add_subplot(gs[0, 2])
    # Show how h evolves for a unit circle
    c_path = np.column_stack([np.cos(t_fine), np.sin(t_fine)])
    p_path, h_path = explorer.compute_strip_sine_prescribed_dynamics(c_path, kappa, alpha)
    
    ax2.plot(t_fine/(2*np.pi), h_path, 'b-', linewidth=2)
    ax2.axhline(y=h_path[-1], color='r', linestyle='--', alpha=0.7, label=f'$\\Delta h = {h_path[-1]:.3f}$')
    ax2.fill_between(t_fine/(2*np.pi), 0, h_path, color='blue', alpha=0.2)
    ax2.set_xlabel('Path parameter (loops)', fontsize=12)
    ax2.set_ylabel('Hidden state $h$', fontsize=12)
    ax2.set_title('Memory Accumulation in Hidden Fiber', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Energy Cost Analysis
    ax3 = fig.add_subplot(gs[1, :])
    radii = np.linspace(0.1, 2.0, 30)
    E_geom, E_prescr, holonomies_R, areas = [], [], [], []
    
    for R in radii:
        c_path = np.column_stack([R*np.cos(t_fine), R*np.sin(t_fine)])
        
        # Geometric lift (h component stays 0)
        p_path_geom, h_path_geom = explorer.compute_strip_sine_geometric_lift(c_path, kappa)
        # Energy functional: E = ∫ ||ů||² dt (squared L2 norm)
        duv_geom = np.diff(p_path_geom, axis=0) / dt  # Velocities
        E_g = np.sum(duv_geom[:, 0]**2 + duv_geom[:, 1]**2) * dt
        E_geom.append(E_g)
        
        # Prescribed dynamics
        p_path_prescr, h_path_prescr = explorer.compute_strip_sine_prescribed_dynamics(c_path, kappa, alpha)
        # Energy functional: E = ∫ ||ů||² dt (squared L2 norm)
        duv = np.diff(p_path_prescr, axis=0) / dt  # Velocities (ů, v̇)
        dh = np.diff(h_path_prescr) / dt  # Velocity ḣ
        # Sum of squared velocities: ů² + v̇² + ḣ²
        E_p = np.sum(duv[:, 0]**2 + duv[:, 1]**2 + dh**2) * dt
        E_prescr.append(E_p)
        
        holonomies_R.append(abs(h_path_prescr[-1]))
        areas.append(np.pi * R**2)  # Area of circle
        
    ax3.plot(radii, E_geom, 'g-', linewidth=2, label='Geometric Lift (ḣ=0)')
    ax3.plot(radii, E_prescr, 'r-', linewidth=2, label='Prescribed Dynamics (with ḣ≠0)')
    ax3.fill_between(radii, E_geom, E_prescr, alpha=0.3, color='orange', label='Excess Cost $\\Delta \\mathcal{E}$')
    
    ax3.set_xlabel('Radius $R$', fontsize=12)
    ax3.set_ylabel('Energy Cost', fontsize=12)
    ax3.set_title('Energy Cost: Geometric vs Prescribed Dynamics', fontsize=14)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(radii, holonomies_R, 'b--', linewidth=2, alpha=0.7, label='Holonomy $|\\Delta h|$')
    ax3_twin.set_ylabel('Holonomy $|\\Delta h|$', fontsize=12, color='blue')
    ax3_twin.tick_params(axis='y', labelcolor='blue')
    ax3_twin.legend(loc='upper right')
    
    # 4. Holonomy vs Area
    ax4 = fig.add_subplot(gs[2, 0])
    # Test different loop sizes centered at origin
    radii_test = np.linspace(0.1, 2.0, 30)
    holonomies_area = []
    areas_test = []
    
    for R in radii_test:
        c_path = np.column_stack([R*np.cos(t_fine), R*np.sin(t_fine)])
        p_path, h_path = explorer.compute_strip_sine_prescribed_dynamics(c_path, kappa, alpha)
        holonomies_area.append(h_path[-1])
        areas_test.append(np.pi * R**2)
    
    # Theory: Δh = 2α·Area
    areas_theory = np.linspace(0, np.pi*4, 100)
    holonomy_theory = 2 * alpha * areas_theory
    
    ax4.plot(areas_test, holonomies_area, 'bo', markersize=6, label='Numerical')
    ax4.plot(areas_theory, holonomy_theory, 'r--', linewidth=2, label='Theory: $\\Delta h = 2\\alpha \\cdot \\mathrm{Area}$')
    ax4.set_xlabel('Enclosed Area', fontsize=12)
    ax4.set_ylabel('Holonomy $\\Delta h$', fontsize=12)
    ax4.set_title('Linear Area-Holonomy Relation', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Memory-Energy Trade-off
    ax5 = fig.add_subplot(gs[2, 1])
    alphas_test = np.linspace(0, 0.6, 25)
    memory_vals, energy_costs = [], []
    R_test = 1.0
    c_path_test = np.column_stack([R_test*np.cos(t_fine), R_test*np.sin(t_fine)])
    area_test = np.pi * R_test**2

    for a in alphas_test:
        # Geometric lift
        p_path_g, h_path_g = explorer.compute_strip_sine_geometric_lift(c_path_test, kappa)
        # Energy functional: E = ∫ ||ů||² dt (squared L2 norm)
        duv_g = np.diff(p_path_g, axis=0) / dt  # Velocities
        E_g = np.sum(duv_g[:, 0]**2 + duv_g[:, 1]**2) * dt
        
        # Prescribed dynamics with varying alpha
        p_path_p, h_path_p = explorer.compute_strip_sine_prescribed_dynamics(c_path_test, kappa, a)
        
        # Energy functional: E = ∫ ||ů||² dt (squared L2 norm)
        duv = np.diff(p_path_p, axis=0) / dt  # (u,v) velocities
        dh = np.diff(h_path_p) / dt  # h velocities
        # Sum of squared velocities
        E_p = np.sum(duv[:, 0]**2 + duv[:, 1]**2 + dh**2) * dt
        
        memory = abs(h_path_p[-1])  # Holonomy in hidden fiber
        memory_vals.append(memory)
        energy_costs.append(E_p - E_g)
    
    # Theory: ΔE = 2π α² R⁴ for a circle of radius R
    memory_theory = 2 * alphas_test * area_test  # = 2α·π·R²
    # For unit circle (R=1): ΔE = 2π α²
    energy_theory = 2 * np.pi * alphas_test**2 * R_test**4
    
    sc = ax5.scatter(memory_vals, energy_costs, c=alphas_test, cmap='viridis', s=50, edgecolor='k', alpha=0.8, label='Numerical')
    ax5.plot(memory_theory, energy_theory, 'r--', linewidth=2, alpha=0.7, label='Theory: $\\Delta\\mathcal{E} \\propto \\alpha^2[\\mathrm{Area}]^2$')
    ax5.set_xlabel('Memory $|\\Delta h|$', fontsize=12)
    ax5.set_ylabel('Excess Energy Cost $\\Delta \\mathcal{E}$', fontsize=12)
    ax5.set_title('Memory-Energy Trade-off', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    cbar = fig.colorbar(sc, ax=ax5)
    cbar.set_label('Gain $\\alpha$', fontsize=10)
    
    # 6. Key Properties Summary
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    props_text = [
        ("",""),
        ('Physical Space:', '$P = \\mathbb{R}^2_{(u,v)} \\times \\mathbb{R}_h$'),
        ('Projection:', '$\\Phi(u,v,h) = (u, v+\\kappa\\sin u)$'),
        ('',''),
        ('Geometric Lift:', '$\\dot{h} = 0$ (no memory)'),
        ('Prescribed:', '$\\dot{h} = \\alpha(c_1\\dot{c}_2 - c_2\\dot{c}_1)$'),
        ('',''),
        ('Holonomy:', '$\\Delta h = 2\\alpha \\cdot \\mathrm{Area}(\\gamma)$'),
        ('Energy Cost:', '$\\Delta\\mathcal{E} \\propto \\alpha^2[\\mathrm{Area}]^2$'),
        ('',''),
        ('Classification:', 'Dynamically Non-Conservative')
    ]
    y_pos = 1.0
    for label, value in props_text:
        ax6.text(0.05, y_pos, label, transform=ax6.transAxes, fontsize=8., fontweight='bold', va='top')
        ax6.text(0.45, y_pos, value, transform=ax6.transAxes, fontsize=8., va='top')
        y_pos -= 0.085

    bg = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.05", facecolor='#FFEBEE', alpha=0.5, edgecolor='#E74C3C', linewidth=1.5, transform=ax6.transAxes)
    ax6.add_patch(bg)
    
    plt.tight_layout()
    return fig

def create_flat_fibration_detailed(explorer):
    """Create detailed figure for flat fibration focusing on zero curvature"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1, 1], width_ratios=[1, 1, 1],
                          hspace=0.3, wspace=0.3)
    
    fig.suptitle('Flat Fibration: Zero Curvature, No Memory', 
                fontsize=16, fontweight='bold')
    
    t = np.linspace(0, 2*np.pi, 200, endpoint=True)
    
    # 1. 3D Bundle Structure
    ax1 = fig.add_subplot(gs[0, :2], projection='3d')
    
    # Base manifold
    theta = np.linspace(0, 2*np.pi, 50)
    r = np.linspace(0, 1.5, 20)
    R, THETA = np.meshgrid(r, theta)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    Z = np.zeros_like(X)
    
    ax1.plot_surface(X, Y, Z, cmap='Blues', edgecolor='none')
    
    # Fibers
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        for radius in [0.5, 1.0, 1.5]:
            x_f = radius * np.cos(angle)
            y_f = radius * np.sin(angle)
            z_fiber = np.linspace(-0.5, 0.5, 20)
            ax1.plot([x_f]*len(z_fiber), [y_f]*len(z_fiber), z_fiber, 
                    'gray', alpha=0.5, linewidth=1)
    
    # Lifted path (closes perfectly)
    x_path = np.cos(t)
    y_path = np.sin(t)
    z_path = np.zeros_like(t)  # No vertical displacement
    
    ax1.plot(x_path, y_path, z_path, 'b-', linewidth=3, label='Lifted path')
    ax1.scatter(x_path[0], y_path[0], z_path[0], color='green', s=100, 
               edgecolor='darkgreen', linewidth=2)
    ax1.scatter(x_path[-1], y_path[-1], z_path[-1], color='red', s=100, 
               edgecolor='darkred', linewidth=2, marker='s')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (fiber)')
    ax1.set_title('Flat Bundle Structure', fontsize=14)
    ax1.view_init(elev=25, azim=45)
    ax1.legend()
    
    # 2. Curvature Visualization
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Show that curvature is zero everywhere
    u = np.linspace(-2, 2, 100)
    v = np.linspace(-2, 2, 100)
    U, V = np.meshgrid(u, v)
    F = np.zeros_like(U)  # Zero curvature
    
    im = ax2.contourf(U, V, F, levels=1, colors=['lightblue'])
    ax2.text(0, 0, 'F = 0', ha='center', va='center', fontsize=20, 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='white'))
    
    ax2.set_xlabel('$u$', fontsize=12)
    ax2.set_ylabel('$v$', fontsize=12)
    ax2.set_title('Curvature Form', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    
    # 3. Parallel Transport
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Show parallel transport around a loop
    n_vectors = 8
    angles = np.linspace(0, 2*np.pi, n_vectors, endpoint=False)
    
    for i, angle in enumerate(angles):
        x = np.cos(angle)
        y = np.sin(angle)
        
        # Vector at this point (remains unchanged)
        vx = 0.2 * np.cos(0)  # Fixed direction
        vy = 0.2 * np.sin(0)
        
        ax3.arrow(x, y, vx, vy, head_width=0.05, head_length=0.03,
                 fc='blue', ec='blue', alpha=0.7)
    
    # Draw the path
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax3.add_patch(circle)
    
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.set_xlabel('X', fontsize=12)
    ax3.set_ylabel('Y', fontsize=12)
    ax3.set_title('Parallel Transport (No Rotation)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. Holonomy Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Show zero holonomy for various loops
    radii = np.linspace(0.5, 2.0, 10)
    holonomies = np.zeros_like(radii)  # All zero
    
    ax4.plot(radii, holonomies, 'b-', linewidth=3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.scatter(radii, holonomies, color='blue', s=50, zorder=5)
    
    ax4.set_xlabel('Loop Radius', fontsize=12)
    ax4.set_ylabel('Holonomy', fontsize=12)
    ax4.set_title('Zero Holonomy for All Loops', fontsize=14)
    ax4.set_ylim(-0.1, 0.1)
    ax4.grid(True, alpha=0.3)
    
    # 5. Energy Analysis
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Show that metric lift = connection lift
    radii = np.linspace(0.1, 2.0, 30)
    E_metric = 2 * np.pi * radii**2  # Squared-speed energy
    E_connection = E_metric  # Same for flat connection
    
    ax5.plot(radii, E_metric, 'g-', linewidth=3, label='Metric/Connection lift')
    ax5.fill_between(radii, 0, E_metric, alpha=0.3, color='green')
    
    ax5.set_xlabel('Radius $R$', fontsize=12)
    ax5.set_ylabel('Energy Cost', fontsize=12)
    ax5.set_title('Minimal Energy Cost', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Phase Space View
    ax6 = fig.add_subplot(gs[2, 0])
    
    # Show closed orbits in phase space
    for R in [0.5, 1.0, 1.5]:
        x = R * np.cos(t)
        y = R * np.sin(t)
        ax6.plot(x, y, 'b-', linewidth=2, alpha=0.7)
    
    ax6.set_xlabel('X', fontsize=12)
    ax6.set_ylabel('Y', fontsize=12)
    ax6.set_title('Closed Orbits', fontsize=14)
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-2, 2)
    ax6.set_ylim(-2, 2)
    
    # 7. Connection Form
    ax7 = fig.add_subplot(gs[2, 1])
    
    # Visualize the trivial connection
    x_grid = np.linspace(-1.5, 1.5, 15)
    y_grid = np.linspace(-1.5, 1.5, 15)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Zero connection means horizontal subspace = tangent to base
    U = -Y_grid / (X_grid**2 + Y_grid**2 + 0.1)  # Circular flow
    V = X_grid / (X_grid**2 + Y_grid**2 + 0.1)
    
    ax7.streamplot(X_grid, Y_grid, U, V, density=1, color='lightgray')
    ax7.set_xlabel('X', fontsize=12)
    ax7.set_ylabel('Y', fontsize=12)
    ax7.set_title('Horizontal Distribution', fontsize=14)
    ax7.set_aspect('equal')
    ax7.set_xlim(-1.5, 1.5)
    ax7.set_ylim(-1.5, 1.5)
    
    # 8. Key Properties
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    props_text = [
        ("",""),
        ('Physical Space:', '$P = \\mathbb{R}^3_{(x,y,z)}$'),
        ('Projection:', '$\\Phi(x,y,z) = (x,y)$'),
        ('',''),
        ('Geometric Lift:', 'Metric lift (minimal energy)'),
        ('Connection:', '$\\omega = dz$ (flat)'),
        ('',''),
        ('Holonomy:', '$\\Delta z = 0$ (all loops)'),
        ('Energy Cost:', '$\\Delta\\mathcal{E} = 0$'),
        ('',''),
        ('Classification:', 'Conditionally Conservative')
    ]
    
    y_pos = 1.0
    for label, value in props_text:
        ax8.text(0.05, y_pos, label, transform=ax8.transAxes, fontsize=8., fontweight='bold', va='top')
        ax8.text(0.45, y_pos, value, transform=ax8.transAxes, fontsize=8., va='top')
        y_pos -= 0.085
    
    bg = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                       boxstyle="round,pad=0.05",
                       facecolor='#E8F5E9', alpha=0.5,
                       edgecolor='#4CAF50', linewidth=2,
                       transform=ax8.transAxes)
    ax8.add_patch(bg)
    
    plt.tight_layout()
    return fig
def compute_hybrid_twisted_lift(c_path, alpha, beta):
    """
    Computes the physical path for a hybrid twisted fibration that produces non-zero holonomy.
    
    This uses a plausible, corrected connection form to match the manuscript's figure:
    ω = dz + (α + β*cos(u))*(y*dx - x*dy)
    """
    p_path_x = c_path[:, 0]
    p_path_y = c_path[:, 1]
    z_path = [0.0]
    
    for i in range(1, len(c_path)):
        x_mid = (p_path_x[i] + p_path_x[i-1]) / 2
        y_mid = (p_path_y[i] + p_path_y[i-1]) / 2
        dx = p_path_x[i] - p_path_x[i-1]
        dy = p_path_y[i] - p_path_y[i-1]
        u_mid = np.arctan2(y_mid, x_mid)
        
        holonomic_term = (y_mid * dx - x_mid * dy)
        dz = -(alpha + beta * np.cos(u_mid)) * holonomic_term
        z_path.append(z_path[-1] + dz)
        
    return np.column_stack([p_path_x, p_path_y, np.array(z_path)])

def create_twisted_fibration_detailed(explorer):
    """
    Creates the final Twisted Fibration figure that matches the manuscript's visual results.
    This version uses a corrected, hybrid connection form to resolve the manuscript's
    internal inconsistency and produce the claimed non-zero holonomy.
    """
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1, 1], width_ratios=[1, 1, 1],
                          hspace=0.45, wspace=0.35)
    
    fig.suptitle('Twisted Fibration: Recreating the Manuscript Figure', 
                fontsize=16, fontweight='bold')
    
    alpha = -0.22
    beta = 0.5
    t = np.linspace(0, 2*np.pi, 201, endpoint=True)
    R = 1.0 
    c_path_unit_circle = np.column_stack([R * np.cos(t), R * np.sin(t)])
    
    # --- Panel 1: 3D Twisted Bundle ---
    ax1 = fig.add_subplot(gs[0, :2], projection='3d')
    p_path = compute_hybrid_twisted_lift(c_path_unit_circle, alpha, beta)
    holonomy_val = p_path[-1, 2] - p_path[0, 2]
    ax1.plot(p_path[:, 0], p_path[:, 1], 0, 'gray', linewidth=1, alpha=0.8, linestyle='--')
    ax1.plot(p_path[:, 0], p_path[:, 1], p_path[:, 2], 'orange', linewidth=3, label='Lifted path')
    ax1.scatter(p_path[0, 0], p_path[0, 1], p_path[0, 2], color='g', s=100, edgecolor='k', zorder=5)
    ax1.scatter(p_path[-1, 0], p_path[-1, 1], p_path[-1, 2], color='r', s=100, edgecolor='k', marker='s', zorder=5)
    ax1.plot([p_path[-1, 0], p_path[-1, 0]], [p_path[-1, 1], p_path[-1, 1]], 
            [p_path[-1, 2], p_path[0, 2]], 'k--', linewidth=2, alpha=0.8, label=f'Holonomy Δz={holonomy_val:.3f}')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z (fiber)')
    ax1.set_title('Correctly Simulated Twisted Bundle', fontsize=14)
    ax1.view_init(elev=25, azim=-50); ax1.legend()

    # --- Panel 2: Variable Curvature Form ---
    ax2 = fig.add_subplot(gs[0, 2])
    u_grid_2d = np.linspace(-np.pi, np.pi, 100)
    r_grid_2d = np.linspace(0, 1.0, 100)
    U_grid_2d, R_grid_2d = np.meshgrid(u_grid_2d, r_grid_2d)
    F_scalar_2d = -2 * (alpha + beta * np.cos(U_grid_2d))
    im = ax2.contourf(U_grid_2d, R_grid_2d, F_scalar_2d, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=ax2, label='$F_{\\rm scalar} \\propto -(\\alpha + \\beta\\cos(u))$')
    ax2.set_xlabel('$\theta$ (angular coordinate)'); ax2.set_ylabel('r (radial coordinate)')
    ax2.set_title('Variable Curvature Form', fontsize=14)
    
    # --- Panel 3: Parallel Transport (FIXED) ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('Parallel Transport (Qualitative)'); ax3.grid(True, alpha=0.3); ax3.set_aspect('equal')
    circle_patch = plt.Circle((0, 0), R, fill=False, edgecolor='black', linewidth=1)
    ax3.add_patch(circle_patch)
    n_vectors = 16
    vector_length = 0.25
    rotation_scaling = 0.4 
    initial_angle = np.pi / 2
    angles_on_circle = np.linspace(0, 2 * np.pi, n_vectors, endpoint=False)
    for i, t_val in enumerate(angles_on_circle):
        x_pos, y_pos = R * np.cos(t_val), R * np.sin(t_val)
        # Rotation is based on the integral of the curvature
        total_rotation = -2 * (alpha * t_val + beta * np.sin(t_val))
        vector_angle = initial_angle + total_rotation * rotation_scaling
        vx, vy = vector_length * np.cos(vector_angle), vector_length * np.sin(vector_angle)
        ax3.arrow(x_pos, y_pos, vx, vy, head_width=0.07, head_length=0.1, fc=plt.cm.viridis(i / n_vectors), ec='k', linewidth=0.5)
    ax3.set_xlim(-1.5, 1.5); ax3.set_ylim(-1.5, 1.5)

    # --- Panel 4: Position-Dependent Holonomy ---
    ax4 = fig.add_subplot(gs[1, 1])
    holonomies_off_center = []
    x_centers = np.linspace(-2, 2, 50)
    for x0 in x_centers:
        c_path_off_center = np.column_stack([x0 + 0.5 * np.cos(t), 0.5 * np.sin(t)])
        p_path_off = compute_hybrid_twisted_lift(c_path_off_center, alpha, beta)
        holonomies_off_center.append(p_path_off[-1, 2])
    ax4.plot(x_centers, holonomies_off_center, 'orange', linewidth=2)
    ax4.set_xlabel('Loop center $x_0$', fontsize=12); ax4.set_ylabel('Holonomy $\\Delta z$', fontsize=12)
    ax4.set_title('Position-Dependent Holonomy', fontsize=14); ax4.grid(True, alpha=0.3)

    # --- Panel 5: Energy Analysis ---
    ax5 = fig.add_subplot(gs[1, 2])
    radii = np.linspace(0.1, 1.5, 20)
    dt = t[1] - t[0]
    
    # Squared-speed energy calculations
    def E_twisted_for_radius(r):
        path = compute_hybrid_twisted_lift(
            np.column_stack([r*np.cos(t), r*np.sin(t)]), alpha, beta
        )
        return energy_squared_speed(path, dt)
    
    E_metric = [energy_squared_speed(
        np.column_stack([r*np.cos(t), r*np.sin(t), np.zeros_like(t)]), dt
    ) for r in radii]
    E_connection = [E_twisted_for_radius(r_val) for r_val in radii]
    ax5.plot(radii, E_metric, 'g-', linewidth=2, label='Metric lift (cost=2πR)')
    ax5.plot(radii, E_connection, 'orange', marker='.', linestyle='-', linewidth=2, label='Connection lift (numerical)')
    ax5.fill_between(radii, E_metric, E_connection, alpha=0.3, color='orange', label='Excess cost $\\Delta\\mathcal{E}$')
    ax5.set_xlabel('Radius R'); ax5.set_ylabel('Energy Cost'); ax5.set_title('Numerically Calculated Energy Cost')
    ax5.legend(); ax5.grid(True, alpha=0.3)

    # --- Panel 6: Fiber Evolution ---
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(t / (2*np.pi), p_path[:, 2], 'orange', linewidth=2)
    ax6.fill_between(t/(2*np.pi), 0, p_path[:, 2], where=(p_path[:,2]<0), color='orange', alpha=0.3)
    ax6.axhline(y=holonomy_val, color='r', linestyle='--', alpha=0.7)
    ax6.set_xlabel('Path parameter (loops)'); ax6.set_ylabel('Z (fiber coordinate)'); ax6.set_title('Fiber Evolution (z vs. t)')
    ax6.grid(True, alpha=0.3)

    # --- Panel 7: Curvature Variation ---
    ax7 = fig.add_subplot(gs[2, 1])
    u_grid_1d = np.linspace(-np.pi, np.pi, 200)
    F_scalar_1d = -2 * (alpha + beta * np.cos(u_grid_1d))
    ax7.plot(u_grid_1d, F_scalar_1d, 'orange', linewidth=2)
    ax7.fill_between(u_grid_1d, 0, F_scalar_1d, where=(F_scalar_1d > 0), alpha=0.3, color='red')
    ax7.fill_between(u_grid_1d, 0, F_scalar_1d, where=(F_scalar_1d < 0), alpha=0.3, color='blue')
    ax7.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax7.set_xlabel('Position $\theta$ (angle)'); ax7.set_ylabel('Curvature Scalar')
    ax7.set_title('Curvature Variation'); ax7.grid(True, alpha=0.3)
    
    # --- Panel 8: Key Properties Summary Box ---
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    props_text = [
        ("",""),
        ('Physical Space:', '$P = \\mathbb{R}^3_{(x,y,z)}$'),
        ('Projection:', '$\\Phi(x,y,z) = (x,y)$'),
        ('',''),
        ('Geometric Lift:', 'Metric lift (minimal)'),
        ('Connection:', '$\\omega = dz + (\\alpha+\\beta\\cos\\theta)(ydx-xdy)$'),
        ('',''),
        ('Holonomy:', '$\\Delta z = 2\\pi\\alpha R^2$ (centered loops)'),
        ('Energy Cost:', '$\\Delta\\mathcal{E} > 0$ (position-dependent)'),
        ('',''),
        ('Classification:', 'Geometrically Non-Conservative')
    ]
    y_pos = 1.0
    for label, value in props_text:
        ax8.text(0.05, y_pos, label, transform=ax8.transAxes, fontsize=8., fontweight='bold', va='top')
        ax8.text(0.45, y_pos, value, transform=ax8.transAxes, fontsize=8., va='top')
        y_pos -= 0.085
    ax8.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.05", facecolor='#FFF3E0', alpha=0.5, edgecolor='#FF9800', linewidth=2, transform=ax8.transAxes))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def create_helical_fibration_detailed(explorer):
    """Create detailed figure for helical fibration"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1, 1], width_ratios=[1, 1, 1],
                          hspace=0.3, wspace=0.3)
    
    fig.suptitle('Helical Fibration: Constant Curvature, Predictable Holonomy', 
                fontsize=16, fontweight='bold')
    
    t = np.linspace(0, 2*np.pi, 200, endpoint=True)
    alpha = 0.3  # Connection parameter
    
    # 1. 3D Helical Structure
    ax1 = fig.add_subplot(gs[0, :2], projection='3d')
    
    # Base manifold
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, 1.5, 20)
    R, THETA = np.meshgrid(r, theta)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    Z = np.zeros_like(X)
    
    ax1.plot_surface(X, Y, Z, alpha=0.2, cmap='Purples', edgecolor='none')
    
    # Vertical fibers
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        for radius in [0.5, 1.0]:
            x_f = radius * np.cos(angle)
            y_f = radius * np.sin(angle)
            z_fiber = np.linspace(0, 2, 20)
            ax1.plot([x_f]*len(z_fiber), [y_f]*len(z_fiber), z_fiber, 
                    'gray', alpha=0.3, linewidth=0.8)
    
    # Helical path
    x_path = np.cos(t)
    y_path = np.sin(t)
    z_path = alpha * t  # Linear rise
    
    ax1.plot(x_path, y_path, z_path, 'purple', linewidth=3, label='Helical lift')
    ax1.scatter(x_path[0], y_path[0], z_path[0], color='green', s=100, 
               edgecolor='darkgreen', linewidth=2)
    ax1.scatter(x_path[-1], y_path[-1], z_path[-1], color='red', s=100, 
               edgecolor='darkred', linewidth=2, marker='s')
    
    # Show holonomy
    ax1.plot([x_path[-1], x_path[-1]], [y_path[-1], y_path[-1]], 
            [z_path[-1], z_path[0]], 'k--', linewidth=2, alpha=0.8)
    ax1.text(x_path[-1], y_path[-1], (z_path[-1] + z_path[0])/2,
            f'Δz = {2*np.pi*alpha:.3f}', fontsize=9.5,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (fiber)')
    ax1.set_title('Helical Bundle Structure', fontsize=14)
    ax1.view_init(elev=20, azim=45)
    ax1.legend()
    
    # 2. Constant Curvature
    ax2 = fig.add_subplot(gs[0, 2])
    
    u = np.linspace(-2, 2, 100)
    v = np.linspace(-2, 2, 100)
    U, V = np.meshgrid(u, v)
    F = 2 * alpha * np.ones_like(U)  # Constant curvature
    
    im = ax2.contourf(U, V, F, levels=1, colors=['purple'], alpha=0.5)
    ax2.text(0, 0, f'F = {2*alpha:.2f}', ha='center', va='center', 
            fontsize=20, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='white'))
    
    ax2.set_xlabel('$u$', fontsize=12)
    ax2.set_ylabel('$v$', fontsize=12)
    ax2.set_title('Constant Curvature Form', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    
    # 3. Linear Holonomy Growth
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Show holonomy vs area
    areas = np.linspace(0, 4*np.pi, 50)
    holonomies = 2 * alpha * areas
    
    ax3.plot(areas/np.pi, holonomies, 'purple', linewidth=3)
    ax3.fill_between(areas/np.pi, 0, holonomies, alpha=0.3, color='purple')
    
    # Mark specific points
    for A in [np.pi, 2*np.pi, 3*np.pi]:
        ax3.scatter(A/np.pi, 2*alpha*A, color='purple', s=100, 
                   edgecolor='black', linewidth=2, zorder=5)
        ax3.text(A/np.pi, 2*alpha*A + 0.2, f'A={A/np.pi:.0f}π', 
                ha='center', fontsize=9)
    
    ax3.set_xlabel('Enclosed Area (units of π)', fontsize=12)
    ax3.set_ylabel('Holonomy Δz', fontsize=12)
    ax3.set_title('Linear Area-Holonomy Relation', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. Parallel Transport
    ax4 = fig.add_subplot(gs[1, 1])
    
    n_vectors = 12
    angles = np.linspace(0, 2*np.pi, n_vectors, endpoint=False)
    
    v0 = np.array([0.2, 0.0])
    
    for i, angle in enumerate(angles):
        x = np.cos(angle)
        y = np.sin(angle)
        
        # Constant rotation rate
        rotation = alpha * angle
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        v_rotated = rotation_matrix @ v0
        
        ax4.arrow(x, y, v_rotated[0], v_rotated[1], 
                 head_width=0.05, head_length=0.03,
                 fc=plt.cm.plasma(i/n_vectors), 
                 ec=plt.cm.plasma(i/n_vectors), alpha=0.8)
    
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax4.add_patch(circle)
    
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_aspect('equal')
    ax4.set_xlabel('X', fontsize=12)
    ax4.set_ylabel('Y', fontsize=12)
    ax4.set_title('Uniform Parallel Transport', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # 5. Energy vs Connection Strength
    ax5 = fig.add_subplot(gs[1, 2])
    
    alphas = np.linspace(0, 0.5, 30)
    R = 1.0  # Unit circle
    dt = t[1] - t[0]
    
    # Squared-speed energy calculations
    def E_from_xyz(x, y, z):
        v = np.column_stack([np.diff(x), np.diff(y), np.diff(z)]) / dt
        return float((v**2).sum(axis=1).sum() * dt)
    
    E_metric = E_from_xyz(np.cos(t), np.sin(t), np.zeros_like(t)) * np.ones_like(alphas)  # = 2π for R=1
    E_connection = np.array([E_from_xyz(np.cos(t), np.sin(t), a*t) for a in alphas])      # = 2π(1+a²) for R=1
    
    ax5.plot(alphas, E_metric, 'g--', linewidth=2, label='Metric lift')
    ax5.plot(alphas, E_connection, 'purple', linewidth=2, label='Connection lift')
    ax5.fill_between(alphas, E_metric, E_connection, alpha=0.3, color='purple')
    
    # Mark our specific alpha with correct squared-speed energy
    specific_energy = E_from_xyz(np.cos(t), np.sin(t), alpha*t)
    ax5.scatter(alpha, specific_energy, 
               color='purple', s=100, edgecolor='black', linewidth=2, zorder=5)
    
    ax5.set_xlabel('Connection parameter α', fontsize=12)
    ax5.set_ylabel('Energy Cost', fontsize=12)
    ax5.set_title('Energy vs Connection Strength', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Helical Projection
    ax6 = fig.add_subplot(gs[2, 0])
    
    # Show the characteristic spiral in 2D
    r = 0.7 + 0.3 * t / (2*np.pi)
    x_spiral = r * np.cos(t)
    y_spiral = r * np.sin(t)
    
    ax6.plot(x_spiral, y_spiral, 'purple', linewidth=3)
    ax6.scatter(x_spiral[0], y_spiral[0], color='green', s=100, 
               edgecolor='darkgreen', linewidth=2)
    ax6.scatter(x_spiral[-1], y_spiral[-1], color='red', s=100, 
               edgecolor='darkred', linewidth=2, marker='s')
    
    # Reference circle
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='gray', 
                       linestyle='--', linewidth=1)
    ax6.add_patch(circle)
    
    ax6.set_xlim(-1.5, 1.5)
    ax6.set_ylim(-1.5, 1.5)
    ax6.set_aspect('equal')
    ax6.set_xlabel('X', fontsize=12)
    ax6.set_ylabel('Y', fontsize=12)
    ax6.set_title('2D Projection (Spiral)', fontsize=14)
    ax6.grid(True, alpha=0.3)
    
    # 7. Memory Accumulation
    ax7 = fig.add_subplot(gs[2, 1])
    
    # Show linear memory accumulation
    time = np.linspace(0, 4*np.pi, 200)
    memory = alpha * time
    
    ax7.plot(time/(2*np.pi), memory, 'purple', linewidth=2)
    ax7.fill_between(time/(2*np.pi), 0, memory, alpha=0.3, color='purple')
    
    # Mark complete loops
    for n in range(1, 3):
        ax7.scatter(n, n*2*np.pi*alpha, color='purple', s=100, 
                   edgecolor='black', linewidth=2, zorder=5)
        ax7.text(n, n*2*np.pi*alpha + 0.3, f'Loop {n}', ha='center', fontsize=9)
    
    ax7.set_xlabel('Number of loops', fontsize=12)
    ax7.set_ylabel('Accumulated memory', fontsize=12)
    ax7.set_title('Linear Memory Growth', fontsize=14)
    ax7.grid(True, alpha=0.3)
    
    # 8. Key Properties
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    props_text = [
        ("",""),
        ('Physical Space:', '$P = \\mathbb{R}^3_{(x,y,z)}$'),
        ('Projection:', '$\\Phi(x,y,z) = (x,y)$'),
        ('',''),
        ('Geometric Lift:', 'Metric lift (α=0)'),
        ('Connection:', '$\\omega = dz - \\alpha(ydx - xdy)$'),
        ('',''),
        ('Holonomy:', '$\\Delta z = 2\\alpha \\cdot \\mathrm{Area}(\\gamma)$'),
        ('Energy Cost:', '$\\Delta\\mathcal{E} > 0$ for $\\alpha > 0$'),
        ('',''),
        ('Classification:', 'Geometrically Non-Conservative')
    ]
    
    y_pos = 1.0
    for label, value in props_text:
        ax8.text(0.05, y_pos, label, transform=ax8.transAxes, fontsize=8., fontweight='bold', va='top')
        ax8.text(0.45, y_pos, value, transform=ax8.transAxes, fontsize=8., va='top')
        y_pos -= 0.085
    
    bg = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                       boxstyle="round,pad=0.05",
                       facecolor='#F3E5F5', alpha=0.5,
                       edgecolor='#9C27B0', linewidth=2,
                       transform=ax8.transAxes)
    ax8.add_patch(bg)
    
    plt.tight_layout()
    return fig

def create_cylinder_fibration_detailed(explorer):
    """Create detailed figure for cylindrical fibration"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1, 1], width_ratios=[1, 1, 1],
                          hspace=0.3, wspace=0.3)
    
    fig.suptitle('Cylindrical Fibration: Flat Connection Despite Non-trivial Topology', 
                fontsize=16, fontweight='bold')
    
    t = np.linspace(0, 2*np.pi, 200, endpoint=True)
    
    # 1. 3D Cylindrical Structure
    ax1 = fig.add_subplot(gs[0, :2], projection='3d')
    
    # Cylinder surface
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(-1, 1, 30)
    THETA, Z = np.meshgrid(theta, z)
    X = np.cos(THETA)
    Y = np.sin(THETA)
    
    ax1.plot_surface(X, Y, Z, alpha=0.3, cmap='Greens', edgecolor='none')
    
    # Fiber circles at different heights
    for z_val in [-0.5, 0, 0.5]:
        x_circle = np.cos(theta)
        y_circle = np.sin(theta)
        z_circle = z_val * np.ones_like(theta)
        ax1.plot(x_circle, y_circle, z_circle, 'gray', alpha=0.5, linewidth=1)
    
    # Path on cylinder
    x_path = np.cos(t)
    y_path = np.sin(t)
    z_path = 0.3 * np.sin(3*t)  # Oscillates but returns to start
    
    ax1.plot(x_path, y_path, z_path, 'green', linewidth=3, label='Path on cylinder')
    ax1.scatter(x_path[0], y_path[0], z_path[0], color='green', s=100, 
               edgecolor='darkgreen', linewidth=2)
    ax1.scatter(x_path[-1], y_path[-1], z_path[-1], color='red', s=100, 
               edgecolor='darkred', linewidth=2, marker='s')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Cylindrical Bundle', fontsize=14)
    ax1.view_init(elev=20, azim=45)
    ax1.legend()
    
    # 2. Zero Curvature
    ax2 = fig.add_subplot(gs[0, 2])
    
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(-1, 1, 100)
    U, V = np.meshgrid(u, v)
    F = np.zeros_like(U)  # Zero curvature despite non-trivial topology
    
    im = ax2.contourf(U, V, F, levels=1, colors=['lightgreen'])
    ax2.text(np.pi, 0, 'F = 0', ha='center', va='center', fontsize=20, 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='white'))
    
    ax2.set_xlabel('$\\theta$', fontsize=12)
    ax2.set_ylabel('$z$', fontsize=12)
    ax2.set_title('Zero Curvature', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 3. Figure-8 Projection
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Show the characteristic figure-8 pattern
    x_fig8 = np.cos(t)
    y_fig8 = np.sin(2*t) * 0.8
    
    ax3.plot(x_fig8, y_fig8, 'green', linewidth=3)
    ax3.scatter(x_fig8[0], y_fig8[0], color='green', s=100, 
               edgecolor='darkgreen', linewidth=2)
    ax3.scatter(x_fig8[-1], y_fig8[-1], color='red', s=100, 
               edgecolor='darkred', linewidth=2, marker='s')
    
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.set_xlabel('X', fontsize=12)
    ax3.set_ylabel('Y', fontsize=12)
    ax3.set_title('Figure-8 Pattern', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. Parallel Transport on Cylinder
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Show that parallel transport preserves direction
    n_vectors = 8
    angles = np.linspace(0, 2*np.pi, n_vectors, endpoint=False)
    
    for i, angle in enumerate(angles):
        x = np.cos(angle)
        y = np.sin(angle)
        
        # Tangent vector (unchanged)
        vx = -0.2 * np.sin(angle)
        vy = 0.2 * np.cos(angle)
        
        ax4.arrow(x, y, vx, vy, head_width=0.05, head_length=0.03,
                 fc='green', ec='green', alpha=0.7)
    
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax4.add_patch(circle)
    
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_aspect('equal')
    ax4.set_xlabel('X', fontsize=12)
    ax4.set_ylabel('Y', fontsize=12)
    ax4.set_title('Trivial Parallel Transport', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # 5. Cost-Memory Analysis (showing zero trade-off)
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Show that holonomy is always zero regardless of loop size
    loop_areas = np.linspace(0, 10, 50)
    holonomy = np.zeros_like(loop_areas)  # Always zero for flat connection
    excess_cost = np.zeros_like(loop_areas)  # No excess cost
    
    ax5.plot(loop_areas, holonomy, 'r-', linewidth=2, label='Holonomy $\\Delta z$')
    ax5.plot(loop_areas, excess_cost, 'b--', linewidth=2, label='Excess cost $\\Delta\\mathcal{E}$')
    
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.set_xlabel('Loop Area', fontsize=12)
    ax5.set_ylabel('Value', fontsize=12)
    ax5.set_title('No Cost-Memory Trade-off', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.5, 1)
    
    # 6. Path Analysis
    ax6 = fig.add_subplot(gs[2, 0])
    
    # Show z-coordinate evolution
    ax6.plot(t/(2*np.pi), z_path, 'green', linewidth=2)
    ax6.fill_between(t/(2*np.pi), 0, z_path, alpha=0.3, color='green')
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax6.set_xlabel('Path parameter (loops)', fontsize=12)
    ax6.set_ylabel('Z coordinate', fontsize=12)
    ax6.set_title('Oscillating but Closed', fontsize=14)
    ax6.grid(True, alpha=0.3)
    
    # 7. Energy Analysis
    ax7 = fig.add_subplot(gs[2, 1])
    
    radii = np.linspace(0.1, 2.0, 30)
    E_metric = 2 * np.pi * radii**2  # Squared-speed energy
    E_connection = E_metric  # For flat connection, same as metric
    
    # Plot both lifts
    ax7.plot(radii, E_metric, 'g-', linewidth=3, 
            label='Metric lift')
    ax7.plot(radii, E_connection, 'b--', linewidth=2, 
            label='Flat connection lift')
    
    # Show they coincide
    ax7.fill_between(radii, 0, E_metric, alpha=0.2, color='green')
    
    # Add theoretical scaling
    ax7.text(0.5, 0.85, '$\\mathcal{E} = 2\\pi R^2$', 
            transform=ax7.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax7.set_xlabel('Radius $R$', fontsize=12)
    ax7.set_ylabel('Energy Cost', fontsize=12)
    ax7.set_title('Energy Analysis: Flat Connection', fontsize=14)
    ax7.legend(loc='upper left')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, 2.0)
    ax7.set_ylim(0, max(E_metric) * 1.1)
    
    # 8. Key Properties
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    props_text = [
        ("",""),
        ('Physical Space:', '$P = (\\mathbb{R}^2\\setminus\\{0\\}) \\times S^1$'),
        ('Projection:', '$\\Phi(x,y,\\theta) = (x,y)$'),
        ('',''),
        ('Geometric Lift:', 'Metric lift (minimal energy)'),
        ('Connection:', '$\\omega = d\\theta$ (flat)'),
        ('',''),
        ('Holonomy:', '$\\Delta\\theta = 0$ (all loops)'),
        ('Energy Cost:', '$\\Delta\\mathcal{E} = 0$'),
        ('',''),
        ('Classification:', 'Conditionally Conservative')
    ]
    
    y_pos = 1.0
    for label, value in props_text:
        ax8.text(0.05, y_pos, label, transform=ax8.transAxes, fontsize=8., fontweight='bold', va='top')
        ax8.text(0.45, y_pos, value, transform=ax8.transAxes, fontsize=8., va='top')
        y_pos -= 0.085
    
    bg = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                       boxstyle="round,pad=0.05",
                       facecolor='#E8F5E9', alpha=0.5,
                       edgecolor='#4CAF50', linewidth=2,
                       transform=ax8.transAxes)
    ax8.add_patch(bg)
    
    plt.tight_layout()
    return fig

# Add methods to explorer
explorer.create_strip_sine_detailed = lambda: create_strip_sine_detailed(explorer)
explorer.create_flat_fibration_detailed = lambda: create_flat_fibration_detailed(explorer)
explorer.create_twisted_fibration_detailed = lambda: create_twisted_fibration_detailed(explorer)
explorer.create_helical_fibration_detailed = lambda: create_helical_fibration_detailed(explorer)
explorer.create_cylinder_fibration_detailed = lambda: create_cylinder_fibration_detailed(explorer)



# Test script to generate detailed figures for each TAS system
import matplotlib.pyplot as plt

print("Generating detailed figures for each TAS system...")
print("These figures focus on curvature and energy aspects as requested.")

# Generate detailed figure for Strip-Sine system
print("\n1. Creating detailed Strip-Sine figure...")
try:
    fig1 = explorer.create_strip_sine_detailed()
    plt.figure(fig1.number)
    plt.savefig('figure_strip_sine_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_strip_sine_detailed.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved as: figure_strip_sine_detailed.pdf/png")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Generate detailed figure for Flat fibration
print("\n2. Creating detailed Flat fibration figure...")
try:
    fig2 = explorer.create_flat_fibration_detailed()
    plt.figure(fig2.number)
    plt.savefig('figure_flat_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_flat_detailed.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved as: figure_flat_detailed.pdf/png")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Generate detailed figure for Cylindrical fibration
print("\n3. Creating detailed Cylindrical fibration figure...")
try:
    fig3 = explorer.create_cylinder_fibration_detailed()
    plt.figure(fig3.number)
    plt.savefig('figure_cylinder_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_cylinder_detailed.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved as: figure_cylinder_detailed.pdf/png")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Generate detailed figure for Twisted fibration
print("\n4. Creating detailed Twisted fibration figure...")
try:
    fig4 = explorer.create_twisted_fibration_detailed()
    plt.figure(fig4.number)
    plt.savefig('figure_twisted_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_twisted_detailed.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved as: figure_twisted_detailed.pdf/png")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Generate detailed figure for Helical fibration
print("\n5. Creating detailed Helical fibration figure...")
try:
    fig5 = explorer.create_helical_fibration_detailed()
    plt.figure(fig5.number)
    plt.savefig('figure_helical_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_helical_detailed.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved as: figure_helical_detailed.pdf/png")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\nAll detailed figures generated!")




plt.show()