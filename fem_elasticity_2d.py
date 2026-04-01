#!/usr/bin/env python3
"""
fem_elasticity_2d.py — FEM solver, plane stress elasticity 2D
==============================================================

Problem identyczny jak PINN (do porównania bezwzględnego błędu):
  Domena: [0,1]², plane stress, E=1.0, ν=0.3
  Elementy: Q4 bilinear quad, całkowanie 2×2 Gauss

Warunki brzegowe (identyczne jak PINN):
  D1: u=0, v=0  w punkcie (0.5, 0.5)   ← blokada translacji
  D2: v=0       w punkcie (1.0, 0.0)   ← blokada obrotu
  Neumann prawa krawędź x=1:
    y ≥ 0.9 → σ_xx=1 (traction), σ_xy=0
    y <  0.9 → σ_xx=0, σ_xy=0  (traction-free)
  Pozostałe krawędzie: traction-free

Wyjście:
  pinn-notebooks/results/fem_solution_{nx}x{ny}.npz
    'u', 'v'    : przemieszczenia na węzłach (n_nodes,)
    'nodes'     : (n_nodes, 2) współrzędne węzłów
    'nx', 'ny'  : rozmiar siatki

Porównanie z PINN reference:
  Po obliczeniu interpoluje FEM → siatka 100×100 PINN
  i liczy L2 error per region (total / load / BC / free)
"""

import os
import time
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata

# ── Stałe fizyczne ────────────────────────────────────────────────
E  = 1.0
NU = 0.3

# Macierz konstytutywna plane stress: D (3×3)
_c = E / (1.0 - NU**2)
D_MAT = _c * np.array([
    [1.0,  NU,        0.0      ],
    [NU,   1.0,       0.0      ],
    [0.0,  0.0,  (1.0-NU)/2.0 ],
])

# Weryfikacja stałych materialnych
assert abs(E - 1.0) < 1e-12
assert abs(NU - 0.3) < 1e-12

# ── Całkowanie Gaussa 2×2 ─────────────────────────────────────────
_GP = np.array([-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)])
_GW = np.array([1.0, 1.0])


# ── Funkcje kształtu Q4 w współrzędnych naturalnych ──────────────
# Kolejność węzłów (CCW od dolnego lewego):
#  n0: (ξ=-1, η=-1)  →  dolny lewy
#  n1: (ξ=+1, η=-1)  →  dolny prawy
#  n2: (ξ=+1, η=+1)  →  górny prawy
#  n3: (ξ=-1, η=+1)  →  górny lewy

def _shape_N(xi, eta):
    """Wartości funkcji kształtu, (4,)."""
    return 0.25 * np.array([
        (1.0-xi)*(1.0-eta),
        (1.0+xi)*(1.0-eta),
        (1.0+xi)*(1.0+eta),
        (1.0-xi)*(1.0+eta),
    ])


def _shape_dN(xi, eta):
    """Pochodne dN/dξ i dN/dη, zwraca (4, 2)."""
    return 0.25 * np.array([
        [-(1.0-eta), -(1.0-xi)],
        [ (1.0-eta), -(1.0+xi)],
        [ (1.0+eta),  (1.0+xi)],
        [-(1.0+eta),  (1.0-xi)],
    ])


# ── Macierz sztywności elementu Q4 ───────────────────────────────

def _element_stiffness(x_e, y_e):
    """
    x_e, y_e: (4,) — współrzędne węzłów elementu
    Zwraca Ke (8×8).
    """
    Ke = np.zeros((8, 8))
    for wi, xi in zip(_GW, _GP):
        for wj, eta in zip(_GW, _GP):
            dN = _shape_dN(xi, eta)                         # (4,2)
            J  = dN.T @ np.column_stack([x_e, y_e])         # (2,2)
            detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
            assert detJ > 0, f"Ujemny detJ={detJ:.3e} — odwrócona kolejność węzłów!"
            dN_xy = dN @ np.linalg.inv(J)                   # (4,2): dN/dx, dN/dy

            # Macierz B (3×8): [dN/dx 0; 0 dN/dy; dN/dy dN/dx] per węzeł
            B = np.zeros((3, 8))
            for n in range(4):
                B[0, 2*n  ] = dN_xy[n, 0]   # dN/dx  → ε_xx
                B[1, 2*n+1] = dN_xy[n, 1]   # dN/dy  → ε_yy
                B[2, 2*n  ] = dN_xy[n, 1]   # dN/dy  → ε_xy (część u)
                B[2, 2*n+1] = dN_xy[n, 0]   # dN/dx  → ε_xy (część v)

            Ke += wi * wj * detJ * (B.T @ D_MAT @ B)
    return Ke


# ── Generacja siatki ──────────────────────────────────────────────

def make_mesh(nx, ny):
    """
    Jednolita siatka Q4 na [0,1]².
    Węzeł (i,j): x=i/nx, y=j/ny, indeks globalny = j*(nx+1)+i

    Zwraca:
      nodes : (n_nodes, 2) — współrzędne (x,y)
      elems : (n_elems, 4) — indeksy węzłów CCW od dolnego lewego
    """
    xs = np.linspace(0.0, 1.0, nx+1)
    ys = np.linspace(0.0, 1.0, ny+1)
    # meshgrid: indeks [j,i] → x=xs[i], y=ys[j]
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])   # row: j*(nx+1)+i

    elems = np.empty((nx*ny, 4), dtype=np.int32)
    idx = 0
    for j in range(ny):
        for i in range(nx):
            n0 = j*(nx+1) + i         # dolny lewy
            n1 = j*(nx+1) + i+1       # dolny prawy
            n2 = (j+1)*(nx+1) + i+1   # górny prawy
            n3 = (j+1)*(nx+1) + i     # górny lewy
            elems[idx] = [n0, n1, n2, n3]
            idx += 1

    return nodes, elems


# ── Montaż globalnej macierzy sztywności ──────────────────────────

def assemble_K(nodes, elems):
    n_dof = 2 * len(nodes)
    K = lil_matrix((n_dof, n_dof))

    for conn in elems:
        x_e = nodes[conn, 0]
        y_e = nodes[conn, 1]
        Ke  = _element_stiffness(x_e, y_e)

        # Indeksy DOF elementu: [2n0, 2n0+1, 2n1, 2n1+1, ...]
        dofs = np.empty(8, dtype=np.int32)
        for k, n in enumerate(conn):
            dofs[2*k  ] = 2*n
            dofs[2*k+1] = 2*n + 1

        rows = np.repeat(dofs, 8)
        cols = np.tile(dofs, 8)
        K[rows, cols] += Ke.ravel()

    return csr_matrix(K)


# ── Wektor sił Neumanna ───────────────────────────────────────────

def assemble_neumann(nodes, nx, ny):
    """
    Prawa krawędź x=1: σ_xx=1 gdzie y≥0.9, σ_xy=0 wszędzie.
    Całkowanie spójne (słaba forma): f_i = ∫_Γ N_i(y) · t(y) dy

    Obsługuje 3 przypadki per element:
      a) element w całości poniżej y=0.9 → brak trakcji
      b) element w całości w strefie załadowanej (y_bot ≥ 0.9) → pełna trakcja
      c) element przekracza y=0.9 (straddling) → ścisłe całkowanie cząstkowe
         (dla ny=100 przypadek (c) nie występuje — y=0.9 leży dokładnie na węźle)
    """
    n_dof = 2 * len(nodes)
    f = np.zeros(n_dof)
    hy = 1.0 / ny
    Y_LOAD = 0.9

    for j in range(ny):
        y_bot = j * hy
        y_top = (j+1) * hy

        n_bot = j*(nx+1) + nx
        n_top = (j+1)*(nx+1) + nx

        if y_bot >= Y_LOAD - 1e-10:
            # Przypadek (b): cały element w strefie załadowanej, tx=1 wszędzie
            # f_bot = hy*(2*1+1)/6 = hy/2,  f_top = hy*(1+2*1)/6 = hy/2
            f[2*n_bot] += hy / 2.0
            f[2*n_top] += hy / 2.0

        elif y_top > Y_LOAD + 1e-10:
            # Przypadek (c): element przekracza y=0.9 (tylko dla niealignowanej siatki)
            # Całkowanie ∫_{y=0.9}^{y_top} N_i(y) * 1 dy
            # Współrzędna naturalna: eta ∈ [-1,1], y = y_bot + (1+eta)/2 * hy
            # y=0.9 → eta_0 = 2*(0.9-y_bot)/hy - 1
            eta_0 = 2.0 * (Y_LOAD - y_bot) / hy - 1.0
            # f_bot = hy/4 * ∫_{eta_0}^{1} (1-eta) deta = hy/4 * [eta - eta²/2]_{eta_0}^{1}
            f[2*n_bot] += hy/4.0 * ((1.0 - 0.5) - (eta_0 - eta_0**2 / 2.0))
            # f_top = hy/4 * ∫_{eta_0}^{1} (1+eta) deta = hy/4 * [eta + eta²/2]_{eta_0}^{1}
            f[2*n_top] += hy/4.0 * ((1.0 + 0.5) - (eta_0 + eta_0**2 / 2.0))

        # Przypadek (a): y_top <= 0.9 → brak trakcji, pomijamy

    return f


# ── Warunki Dirichleta (eliminacja bezpośrednia) ──────────────────

def apply_dirichlet(K_csr, f, nodes, nx, ny):
    """
    D1: u=0, v=0 w (0.5, 0.5) — blokuje translację x i y
    D2: v=0      w (1.0, 0.0) — blokuje obrót

    Metoda eliminacji: zerowanie wiersza/kolumny, 1 na diagonali.
    """
    K = K_csr.tolil()

    # Wyznacz węzły Dirichleta
    # D1: węzeł (i=nx//2, j=ny//2) dla parzystych nx, ny
    i_d1, j_d1 = nx//2, ny//2
    n_d1 = j_d1*(nx+1) + i_d1
    assert abs(nodes[n_d1, 0] - 0.5) < 1e-10 and abs(nodes[n_d1, 1] - 0.5) < 1e-10, \
        f"D1 węzeł {n_d1} ma współrzędne {nodes[n_d1]} ≠ (0.5, 0.5)"

    # D2: węzeł (i=nx, j=0)
    n_d2 = 0*(nx+1) + nx
    assert abs(nodes[n_d2, 0] - 1.0) < 1e-10 and abs(nodes[n_d2, 1] - 0.0) < 1e-10, \
        f"D2 węzeł {n_d2} ma współrzędne {nodes[n_d2]} ≠ (1.0, 0.0)"

    # Lista (dof, wartość)
    constrained = [
        (2*n_d1,     0.0),   # D1: u=0
        (2*n_d1 + 1, 0.0),   # D1: v=0
        (2*n_d2 + 1, 0.0),   # D2: v=0
    ]

    for dof, val in constrained:
        # Korekta RHS: f -= K[:,dof] * val  (val=0 → no-op, ale zostawiam dla ogólności)
        col = np.array(K[:, dof].todense()).ravel()
        f -= col * val
        # Zerowanie wiersza i kolumny
        K[dof, :] = 0.0
        K[:, dof] = 0.0
        # Jedynka na diagonali
        K[dof, dof] = 1.0
        f[dof] = val

    return csr_matrix(K), f


# ── Metryki L2 per region ─────────────────────────────────────────

def compute_l2_metrics(u_fem, v_fem, u_ref, v_ref, grid):
    """
    Oblicza L2 error per region (definicje identyczne jak w PINN benchmark).
    grid: (N, 2)
    """
    x, y = grid[:, 0], grid[:, 1]

    mask_load = (x >= 0.9) & (y >= 0.9)
    mask_bc   = (
        (np.sqrt((x-0.5)**2 + (y-0.5)**2) < 0.1)
      | (np.sqrt((x-1.0)**2 + (y-0.0)**2) < 0.1)
    )
    mask_free = ~mask_load & ~mask_bc

    def l2(err, mask):
        return float(np.sqrt(np.mean(err[mask]**2))) if mask.sum() > 0 else float('nan')

    err_u = u_fem - u_ref
    err_v = v_fem - v_ref

    return {
        'l2u_total': float(np.sqrt(np.mean(err_u**2))),
        'l2v_total': float(np.sqrt(np.mean(err_v**2))),
        'l2u_load':  l2(err_u, mask_load),
        'l2v_load':  l2(err_v, mask_load),
        'l2u_bc':    l2(err_u, mask_bc),
        'l2v_bc':    l2(err_v, mask_bc),
        'l2u_free':  l2(err_u, mask_free),
        'l2v_free':  l2(err_v, mask_free),
        'n_load':    int(mask_load.sum()),
        'n_bc':      int(mask_bc.sum()),
        'n_free':    int(mask_free.sum()),
    }


# ── Główna funkcja solvera ────────────────────────────────────────

def solve_fem(nx=100, ny=100, save_path=None, compare_pinn_ref=True):
    """
    Uruchamia pełny pipeline FEM.

    Parametry:
      nx, ny          : liczba elementów w x i y
      save_path       : ścieżka zapisu .npz (None = nie zapisuj)
      compare_pinn_ref: czy porównać z reference_solution.npz

    Zwraca:
      nodes, u, v — wyniki FEM
    """
    n_nodes = (nx+1) * (ny+1)
    n_dof   = 2 * n_nodes
    n_elems = nx * ny

    print("=" * 60)
    print(f"  FEM solver Q4 — plane stress — E={E}, ν={NU}")
    print(f"  Siatka: {nx}×{ny} elementów")
    print(f"  Węzły: {n_nodes}   DOF: {n_dof}   Elementy: {n_elems}")
    print("=" * 60)

    t0 = time.perf_counter()

    # 1. Siatka
    print("\n[1/4] Generacja siatki...")
    nodes, elems = make_mesh(nx, ny)
    print(f"      x ∈ [{nodes[:,0].min():.3f}, {nodes[:,0].max():.3f}]"
          f"  y ∈ [{nodes[:,1].min():.3f}, {nodes[:,1].max():.3f}]")

    # 2. Montaż K
    print("[2/4] Montaż macierzy sztywności...")
    t1 = time.perf_counter()
    K = assemble_K(nodes, elems)
    print(f"      Czas: {time.perf_counter()-t1:.2f}s  "
          f"  nnz(K) = {K.nnz}  "
          f"  wypełnienie = {K.nnz/n_dof**2*100:.4f}%")

    # 3. Warunki brzegowe
    print("[3/4] Warunki brzegowe...")
    f = assemble_neumann(nodes, nx, ny)
    print(f"      Całkowita siła Neumanna (Fx): {f[0::2].sum():.6f}  (oczekiwane: 0.1)")
    K, f = apply_dirichlet(K, f, nodes, nx, ny)

    # 4. Rozwiązanie
    print("[4/4] Rozwiązanie układu (spsolve)...")
    t2 = time.perf_counter()
    u_all = spsolve(K, f)
    print(f"      Czas solve: {time.perf_counter()-t2:.3f}s")

    u = u_all[0::2]
    v = u_all[1::2]

    elapsed = time.perf_counter() - t0
    print(f"\n  u: [{u.min():.4e}, {u.max():.4e}]  (max={u.max():.4e})")
    print(f"  v: [{v.min():.4e}, {v.max():.4e}]")
    print(f"  Całkowity czas: {elapsed:.2f}s")

    # Zapis
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path,
                 u=u, v=v, nodes=nodes,
                 nx=np.int32(nx), ny=np.int32(ny),
                 E=np.float64(E), nu=np.float64(NU),
                 elapsed_s=np.float64(elapsed))
        print(f"\n  Zapisano: {save_path}")

    # Porównanie z PINN reference
    if compare_pinn_ref:
        ref_path = "pinn-notebooks/results/reference_solution.npz"
        if not os.path.exists(ref_path):
            print(f"\n  ⚠ Brak {ref_path} — pomijam porównanie z PINN")
        else:
            print("\n" + "─"*60)
            print("  Porównanie FEM vs Reference PINN (N=10k)")
            print("─"*60)

            ref  = np.load(ref_path)
            grid = ref['grid'].astype(np.float64)    # (10000, 2)
            u_ref = ref['u'].astype(np.float64)
            v_ref = ref['v'].astype(np.float64)

            # Interpolacja FEM → siatka PINN (bilinear)
            print("  Interpolacja FEM → siatka 100×100...")
            u_fem_interp = griddata(nodes, u, grid, method='linear')
            v_fem_interp = griddata(nodes, v, grid, method='linear')

            # Sprawdź NaN (punkty poza domeną interpolacji)
            nan_mask = np.isnan(u_fem_interp)
            if nan_mask.any():
                print(f"  ⚠ {nan_mask.sum()} punktów poza zakresem interpolacji — uzupełniam nearest")
                u_fem_nn = griddata(nodes, u, grid, method='nearest')
                v_fem_nn = griddata(nodes, v, grid, method='nearest')
                u_fem_interp[nan_mask] = u_fem_nn[nan_mask]
                v_fem_interp[nan_mask] = v_fem_nn[nan_mask]

            # L2 per region
            m = compute_l2_metrics(u_fem_interp, v_fem_interp, u_ref, v_ref, grid)

            l2_u_ref_norm = float(np.sqrt(np.mean(u_ref**2)))
            l2_v_ref_norm = float(np.sqrt(np.mean(v_ref**2)))

            print(f"\n  {'Region':<12} {'L2(u)':<12} {'L2(v)':<12} {'Wzgl. u':<12}")
            print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
            regions = [('total', 'l2u_total', 'l2v_total'),
                       ('load',  'l2u_load',  'l2v_load'),
                       ('BC',    'l2u_bc',    'l2v_bc'),
                       ('free',  'l2u_free',  'l2v_free')]
            for name, ku, kv in regions:
                pct = m[ku] / l2_u_ref_norm * 100 if l2_u_ref_norm > 0 else float('nan')
                print(f"  {name:<12} {m[ku]:<12.4e} {m[kv]:<12.4e} {pct:<11.2f}%")

            # Dla kontekstu: GIS vs reference PINN (z tabeli AGENTS.md)
            print(f"\n  Dla kontekstu (z benchmark):")
            print(f"    GIS  vs ref-PINN: L2u_total=4.05e-03  (~11% wzgl.)")
            print(f"    Sobol vs ref-PINN: L2u_total=3.69e-02  (~100% wzgl.)")

            if save_path:
                comp_path = save_path.replace('.npz', '_vs_pinn_ref.npz')
                np.savez(comp_path,
                         u_fem_interp=u_fem_interp,
                         v_fem_interp=v_fem_interp,
                         grid=grid,
                         **{k: np.float64(v_) for k, v_ in m.items()
                            if isinstance(v_, float)})
                print(f"\n  Wyniki porównania: {comp_path}")

    return nodes, u, v


# ── Weryfikacja całkowitej siły ───────────────────────────────────

def verify_total_force(nx=100, ny=100):
    """
    Szybka weryfikacja: całkowita siła Neumanna na prawej krawędzi
    powinna wynosić Fx = 1.0 × (1.0 - 0.9) = 0.1
    """
    nodes, _ = make_mesh(nx, ny)
    f = assemble_neumann(nodes, nx, ny)
    Fx_total = f[0::2].sum()
    print(f"Całkowita siła Fx = {Fx_total:.10f}  (oczekiwane: 0.1000000000)")
    assert abs(Fx_total - 0.1) < 1e-9, f"Błąd! Fx={Fx_total:.10f}"
    print("OK ✓")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Weryfikacja sił Neumanna przed głównym rozwiązaniem
    print("Weryfikacja sił Neumanna...")
    verify_total_force(nx=100, ny=100)
    print()

    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else nx

    save_path = f"pinn-notebooks/results/fem_solution_{nx}x{ny}.npz"

    nodes, u, v = solve_fem(
        nx=nx, ny=ny,
        save_path=save_path,
        compare_pinn_ref=True,
    )
