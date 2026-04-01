#!/usr/bin/env python3
"""
fem_highres.py — High-resolution FEM reference solution for PINN comparison
============================================================================

Uruchamia istniejący solver FEM (fem_elasticity_2d.py) na gęstej siatce
i zapisuje KOMPLET danych do zestawiania z PINN:

Wyjście: pinn-notebooks/results/fem_ground_truth_{nx}x{ny}.npz

  Na węzłach FEM (high-res):
    nodes_xy    : (n_nodes, 2)  — współrzędne węzłów
    u_nodes     : (n_nodes,)    — przemieszczenie u
    v_nodes     : (n_nodes,)    — przemieszczenie v

  Na standardowej siatce PINN 100×100 = 10 000 punktów:
    grid_xy     : (10000, 2)    — punkty ewaluacji (identyczne jak PINN)
    u_grid      : (10000,)      — u interpolowane z FEM → ground truth
    v_grid      : (10000,)      — v interpolowane z FEM → ground truth
    sigma_xx_grid : (10000,)    — σ_xx interpolowane z centroidów elementów
    sigma_yy_grid : (10000,)    — σ_yy
    sigma_xy_grid : (10000,)    — σ_xy

  W centroidach elementów:
    centroids_xy: (n_elems, 2)  — środki ciężkości elementów
    sigma_xx    : (n_elems,)    — naprężenie σ_xx
    sigma_yy    : (n_elems,)    — naprężenie σ_yy
    sigma_xy    : (n_elems,)    — naprężenie σ_xy

  Metadane:
    nx, ny, E, nu, elapsed_s

Jak porównywać z PINN:
    ref = np.load('pinn-notebooks/results/fem_ground_truth_1000x1000.npz')
    u_gt = ref['u_grid']    # (10000,) — ground truth u na siatce PINN
    v_gt = ref['v_grid']    # (10000,) — ground truth v
    grid = ref['grid_xy']   # (10000, 2) — punkty ewaluacji

    # Po treningu PINN:
    with torch.no_grad():
        uv = model(torch.tensor(grid, dtype=torch.float32)).numpy()
    l2_u = np.sqrt(np.mean((uv[:,0] - u_gt)**2))
"""

import os
import sys
import time
import numpy as np
from scipy.interpolate import griddata
from scipy.sparse import coo_matrix, csr_matrix

# ── Import z istniejącego, zweryfikowanego solvera ────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fem_elasticity_2d import (
    solve_fem, make_mesh, verify_total_force,
    assemble_neumann, apply_dirichlet,
    D_MAT, _shape_dN, _element_stiffness, E, NU
)

RESULTS_DIR = "pinn-notebooks/results"
PINN_REF_PATH = f"{RESULTS_DIR}/reference_solution.npz"


# ── Wektoryzowany assembler macierzy sztywności ───────────────────

def assemble_K_fast(nodes, elems):
    """
    Wektoryzowany montaż globalnej macierzy sztywności K dla jednolitej siatki Q4.

    Dla siatki prostokątnej wszystkie elementy mają IDENTYCZNE Ke (ten sam hx, hy),
    więc wystarczy obliczyć Ke raz i zwielokrotnić indeksy DOF przez numpy.

    Złożoność:  O(n_elems) operacji numpy (vs O(n_elems) iteracji Python w wersji naiwnej).
    Szybkość:   ~100× szybsza niż oryginalna pętla Python dla n_elems = 1M.

    Pamięć:     ~1 GB pośrednich danych COO dla 1000×1000 (64M wpisów int32/float64).
    """
    n_dof   = 2 * len(nodes)
    n_elems = len(elems)

    # --- Ke takie samo dla każdego elementu ---
    e0 = elems[0]
    hx = nodes[e0[1], 0] - nodes[e0[0], 0]   # szerokość elementu
    hy = nodes[e0[3], 1] - nodes[e0[0], 1]   # wysokość elementu
    Ke = _element_stiffness(
        np.array([0.0, hx, hx, 0.0]),
        np.array([0.0, 0.0, hy, hy])
    )                                          # (8, 8)
    Ke_flat = Ke.ravel()                       # (64,)

    # --- DOF globalny dla każdego elementu: (n_elems, 8) ---
    eidofs = np.empty((n_elems, 8), dtype=np.int32)
    for k in range(4):
        eidofs[:, 2*k  ] = (2 * elems[:, k]).astype(np.int32)
        eidofs[:, 2*k+1] = (2 * elems[:, k] + 1).astype(np.int32)

    # --- Wektory COO (n_elems × 64) ---
    # Ke[i, j] trafia do K[eidof[i], eidof[j]].
    # Ke.ravel() (row-major): indeks k → i = k//8, j = k%8
    #   rows[k] = eidof[i] = np.repeat(eidofs, 8, axis=1)[k]   ✓
    #   cols[k] = eidof[j] = np.tile(eidofs, (1, 8))[k]        ✓
    CHUNK = 200_000   # ~160 MB szczytowo per chunk (int32 + float64)
    all_rows, all_cols, all_vals = [], [], []

    for start in range(0, n_elems, CHUNK):
        end  = min(start + CHUNK, n_elems)
        sub  = eidofs[start:end]                       # (sz, 8)
        r    = np.repeat(sub, 8, axis=1).ravel()       # (sz*64,) int32
        c    = np.tile(sub, (1, 8)).ravel()            # (sz*64,) int32
        v    = np.tile(Ke_flat, end - start)           # (sz*64,) float64
        all_rows.append(r)
        all_cols.append(c)
        all_vals.append(v)

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.concatenate(all_vals)

    K = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return K.tocsr()


# ── Solver układu (z awaryjnym CG+ILU) ───────────────────────────

def solve_system(K_csr, f):
    """
    Rozwiązuje K @ u = f. Próbuje SuperLU (spsolve), jeśli nie − CG+ILU.
    Dla n_dof ~ 2M spsolve powinien działać w < 10 min z ~2–4 GB RAM.
    """
    from scipy.sparse.linalg import spsolve
    try:
        print("      Próba: spsolve (SuperLU/UMFPACK)...")
        u = spsolve(K_csr, f)
        if np.isnan(u).any() or np.isinf(u).any():
            raise ValueError("spsolve zwrócił NaN/Inf")
        return u
    except Exception as e:
        print(f"      spsolve nieudany ({e}), przejście na CG + ILU...")
        from scipy.sparse.linalg import cg, LinearOperator
        from scipy.sparse.linalg import spilu

        ilu = spilu(K_csr.astype(np.float64), fill_factor=10)
        M   = LinearOperator(K_csr.shape, ilu.solve)
        u, info = cg(K_csr, f, M=M, maxiter=5000, tol=1e-10)
        if info != 0:
            print(f"      ⚠ CG nie zbiegł (info={info}) — wyniki mogą być niedokładne")
        return u


# ── Naprężenia w centroidach elementów ───────────────────────────

def compute_stresses(nodes, elems, u, v):
    """
    Oblicza [σ_xx, σ_yy, σ_xy] w centroidach elementów Q4 (xi=0, eta=0).

    Dla jednolitej siatki prostokątnej Jakobian jest taki sam dla każdego
    elementu — obliczamy B raz, stosujemy do wszystkich (numpy broadcasting).

    Zwraca:
      centroids : (n_elems, 2)
      sigma_xx  : (n_elems,)
      sigma_yy  : (n_elems,)
      sigma_xy  : (n_elems,)
    """
    n_elems = len(elems)

    # Centroidy = średnia współrzędnych 4 węzłów elementu
    centroids = nodes[elems].mean(axis=1)  # (n_elems, 2)

    # Rozmiar elementu (jednolita siatka):
    hx = nodes[elems[0, 1], 0] - nodes[elems[0, 0], 0]
    hy = nodes[elems[0, 3], 1] - nodes[elems[0, 0], 1]

    # Pochodne f. kształtu w centroidzie (xi=0, eta=0): (4, 2)
    dN_nat = _shape_dN(0.0, 0.0)

    # Jakobian prostokątny: J = diag(hx/2, hy/2) → J^{-1} = diag(2/hx, 2/hy)
    dN_xy = dN_nat * np.array([2.0/hx, 2.0/hy])  # (4, 2) dN/dx, dN/dy

    # Macierz B (3×8) — tożsama dla wszystkich elementów jednolitej siatki
    B = np.zeros((3, 8))
    for n in range(4):
        B[0, 2*n  ] = dN_xy[n, 0]  # ε_xx ← dNn/dx
        B[1, 2*n+1] = dN_xy[n, 1]  # ε_yy ← dNn/dy
        B[2, 2*n  ] = dN_xy[n, 1]  # ε_xy ← dNn/dy (u-część)
        B[2, 2*n+1] = dN_xy[n, 0]  # ε_xy ← dNn/dx (v-część)

    # Wektor przemieszczeń per element: (n_elems, 8)
    # [u_n0, v_n0, u_n1, v_n1, u_n2, v_n2, u_n3, v_n3]
    d_e = np.empty((n_elems, 8))
    for k in range(4):
        d_e[:, 2*k  ] = u[elems[:, k]]
        d_e[:, 2*k+1] = v[elems[:, k]]

    # σ = D @ B @ d_e  → (n_elems, 3) przez broadcasting
    DB = D_MAT @ B           # (3, 8)
    sigma = d_e @ DB.T       # (n_elems, 3)

    return centroids, sigma[:, 0], sigma[:, 1], sigma[:, 2]


# ── Główny pipeline ───────────────────────────────────────────────

def run_highres_fem(nx=1000, ny=1000):
    """
    Uruchamia FEM nx×ny i zapisuje KOMPLET danych do porównania z PINN.

    Używa wektoryzowanego asemblera K (assemble_K_fast) — ~100× szybszy
    niż oryginalna pętla Python dla siatek nx×ny > 100×100.

    Zapisuje do: pinn-notebooks/results/fem_ground_truth_{nx}x{ny}.npz
    """
    out_path = f"{RESULTS_DIR}/fem_ground_truth_{nx}x{ny}.npz"

    print("=" * 65)
    print(f"  FEM HIGH-RES REFERENCE — {nx}×{ny} elementów")
    print(f"  Węzły: {(nx+1)*(ny+1):,}   DOF: {2*(nx+1)*(ny+1):,}")
    print(f"  Wyjście: {out_path}")
    print("=" * 65)

    t_start = time.perf_counter()

    # ── Krok 1: Siatka + weryfikacja sił Neumanna ─────────────────
    print("\n[1/6] Generacja siatki + weryfikacja BC...")
    nodes, elems = make_mesh(nx, ny)
    f_neu = assemble_neumann(nodes, nx, ny)
    Fx = float(f_neu[0::2].sum())
    Fy = float(f_neu[1::2].sum())
    print(f"      Siła Neumanna Fx = {Fx:.10f}  (oczekiwane: 0.1000000000)")
    print(f"      Siła Neumanna Fy = {Fy:.10f}  (oczekiwane: 0.0000000000)")
    assert abs(Fx - 0.1) < 1e-9, f"BŁĄD! Fx={Fx}"

    # ── Krok 2: Montaż K (wektoryzowany) ─────────────────────────
    print(f"\n[2/6] Montaż K — wektoryzowany ({nx*ny:,} elementów)...")
    t2 = time.perf_counter()
    K = assemble_K_fast(nodes, elems)
    print(f"      Czas: {time.perf_counter()-t2:.1f}s   nnz(K)={K.nnz:,}")

    # ── Krok 3: Dirichlet BC + solve ─────────────────────────────
    print("\n[3/6] Dirichlet BC + rozwiązanie układu...")
    f_vec = f_neu.copy()
    K_mod, f_mod = apply_dirichlet(K, f_vec, nodes, nx, ny)
    t3 = time.perf_counter()
    u_all = solve_system(K_mod, f_mod)
    print(f"      Czas solve: {time.perf_counter()-t3:.1f}s")

    u = u_all[0::2]
    v = u_all[1::2]
    print(f"      u ∈ [{u.min():.4e}, {u.max():.4e}]")
    print(f"      v ∈ [{v.min():.4e}, {v.max():.4e}]")

    # Energia odkształcenia: U = 0.5 * W_zew = 0.5 * f_neu · u_all
    u_all_full = np.zeros(2 * len(nodes), dtype=np.float64)
    u_all_full[0::2] = u
    u_all_full[1::2] = v
    energy_strain = 0.5 * float(f_neu @ u_all_full)
    print(f"      Energia odkształcenia U = {energy_strain:.6e}")

    # ── Krok 4: Naprężenia w centroidach ──────────────────────────
    print("\n[4/6] Naprężenia w centroidach elementów...")
    t4 = time.perf_counter()
    centroids, sigma_xx, sigma_yy, sigma_xy = compute_stresses(nodes, elems, u, v)
    # Von Mises (plane stress): σ_VM = √(σ_xx² - σ_xx·σ_yy + σ_yy² + 3·σ_xy²)
    von_mises = np.sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2
                        + 3.0*sigma_xy**2)
    print(f"      Czas: {time.perf_counter()-t4:.2f}s")
    print(f"      σ_xx  ∈ [{sigma_xx.min():.4e}, {sigma_xx.max():.4e}]")
    print(f"      σ_yy  ∈ [{sigma_yy.min():.4e}, {sigma_yy.max():.4e}]")
    print(f"      σ_xy  ∈ [{sigma_xy.min():.4e}, {sigma_xy.max():.4e}]")
    print(f"      σ_VM max = {von_mises.max():.4e}")

    mask_load_c = (centroids[:, 0] > 0.99) & (centroids[:, 1] >= 0.9)
    if mask_load_c.sum() > 0:
        print(f"      σ_xx przy x≈1, y≥0.9: śr={sigma_xx[mask_load_c].mean():.4f}"
              f"  (oczekiwane ≈ 1.0 na krawędzi)")

    # ── Krok 5: Interpolacja → siatka PINN 100×100 ───────────────
    print("\n[5/6] Interpolacja FEM → siatka PINN 100×100 (10 000 pkt)...")
    t5 = time.perf_counter()

    if os.path.exists(PINN_REF_PATH):
        ref_npz = np.load(PINN_REF_PATH)
        grid_xy = ref_npz['grid'].astype(np.float64)   # (10000, 2)
        print(f"      Załadowano siatkę z {PINN_REF_PATH}")
    else:
        xs = np.linspace(0.0, 1.0, 100)
        ys = np.linspace(0.0, 1.0, 100)
        xx, yy = np.meshgrid(xs, ys)
        grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
        print(f"      ⚠ Brak {PINN_REF_PATH} — standardowa siatka 100×100")

    def interp(src_pts, src_vals, dst_pts, label):
        arr = griddata(src_pts, src_vals, dst_pts, method='linear')
        n_nan = int(np.isnan(arr).sum())
        if n_nan > 0:
            print(f"      ⚠ {label}: {n_nan} NaN → nearest neighbour")
            mask_nan = np.isnan(arr)
            arr[mask_nan] = griddata(
                src_pts, src_vals, dst_pts[mask_nan], method='nearest')
        return arr

    u_grid    = interp(nodes,     u,         grid_xy, 'u_grid')
    v_grid    = interp(nodes,     v,         grid_xy, 'v_grid')
    sxx_grid  = interp(centroids, sigma_xx,  grid_xy, 'sigma_xx_grid')
    syy_grid  = interp(centroids, sigma_yy,  grid_xy, 'sigma_yy_grid')
    sxy_grid  = interp(centroids, sigma_xy,  grid_xy, 'sigma_xy_grid')
    vm_grid   = interp(centroids, von_mises, grid_xy, 'von_mises_grid')

    print(f"      Czas: {time.perf_counter()-t5:.2f}s")
    print(f"      u_grid  ∈ [{u_grid.min():.4e}, {u_grid.max():.4e}]")
    print(f"      v_grid  ∈ [{v_grid.min():.4e}, {v_grid.max():.4e}]")

    # Region masks na siatce PINN — definicje z AGENTS.md (niezmienne)
    x_g, y_g = grid_xy[:, 0], grid_xy[:, 1]
    mask_load = (x_g >= 0.9) & (y_g >= 0.9)
    mask_bc   = (
        (np.sqrt((x_g - 0.5)**2 + (y_g - 0.5)**2) < 0.1) |
        (np.sqrt((x_g - 1.0)**2 + (y_g - 0.0)**2) < 0.1)
    )
    mask_free = ~mask_load & ~mask_bc
    print(f"      load={mask_load.sum()} BC={mask_bc.sum()} "
          f"free={mask_free.sum()} total={len(x_g)}")

    # Porównanie z PINN reference (jeśli dostępny)
    pinn_metrics = {}
    if os.path.exists(PINN_REF_PATH):
        u_pinn = ref_npz['u'].astype(np.float64)
        v_pinn = ref_npz['v'].astype(np.float64)
        err_u  = u_grid - u_pinn
        err_v  = v_grid - v_pinn
        u_rms  = float(np.sqrt(np.mean(u_pinn**2)))
        print(f"\n      FEM vs PINN reference (u_rms_ref={u_rms:.4e}):")
        print(f"      {'Region':<10} {'L2(u)':<12} {'L2(v)':<12} {'wzgl. u'}")
        for name, mask in [
            ('total', np.ones(len(x_g), dtype=bool)),
            ('load',  mask_load),
            ('BC',    mask_bc),
            ('free',  mask_free),
        ]:
            lu  = float(np.sqrt(np.mean(err_u[mask]**2)))
            lv  = float(np.sqrt(np.mean(err_v[mask]**2)))
            pct = lu / u_rms * 100 if u_rms > 0 else float('nan')
            print(f"      {name:<10} {lu:<12.4e} {lv:<12.4e} {pct:.1f}%")
            pinn_metrics[f'vs_pinn_l2u_{name}'] = np.float64(lu)
            pinn_metrics[f'vs_pinn_l2v_{name}'] = np.float64(lv)

    # ── Krok 6: Zapis ─────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print(f"\n[6/6] Zapis {out_path}...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    np.savez(
        out_path,
        # ── Węzły FEM (high-res) ──────────────────────────────────
        nodes_xy       = nodes,
        u_nodes        = u,
        v_nodes        = v,
        # ── Naprężenia + Von Mises w centroidach ──────────────────
        centroids_xy   = centroids,
        sigma_xx       = sigma_xx,
        sigma_yy       = sigma_yy,
        sigma_xy       = sigma_xy,
        von_mises      = von_mises,
        # ── Siatka PINN 100×100 — pola przemieszczeń ─────────────
        grid_xy        = grid_xy,
        u_grid         = u_grid,
        v_grid         = v_grid,
        # ── Siatka PINN 100×100 — pola naprężeń ──────────────────
        sigma_xx_grid  = sxx_grid,
        sigma_yy_grid  = syy_grid,
        sigma_xy_grid  = sxy_grid,
        von_mises_grid = vm_grid,
        # ── Region masks (definicje z AGENTS.md) ─────────────────
        mask_load      = mask_load,
        mask_bc        = mask_bc,
        mask_free      = mask_free,
        # ── Weryfikacja fizyczna ──────────────────────────────────
        force_Fx       = np.float64(Fx),
        force_Fy       = np.float64(Fy),
        energy_strain  = np.float64(energy_strain),
        # ── Metadane ──────────────────────────────────────────────
        nx             = np.int32(nx),
        ny             = np.int32(ny),
        E              = np.float64(E),
        nu             = np.float64(NU),
        elapsed_s      = np.float64(elapsed),
        **pinn_metrics,
    )

    print(f"\n{'='*65}")
    print(f"  GOTOWE!  Czas: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Zapisano: {out_path}")
    print(f"  u ∈ [{u.min():.4e}, {u.max():.4e}]   "
          f"v ∈ [{v.min():.4e}, {v.max():.4e}]")
    print(f"  σ_VM max = {von_mises.max():.4e}   "
          f"U_strain = {energy_strain:.4e}")
    print(f"  Weryfikacja: Fx={Fx:.8f}  Fy={Fy:.8f}")
    print(f"\n  Jak używać do porównania z PINN:")
    print(f"    gt    = np.load('{out_path}', allow_pickle=True)")
    print(f"    u_gt  = gt['u_grid']       # (10000,)  — ground truth u")
    print(f"    v_gt  = gt['v_grid']       # (10000,)  — ground truth v")
    print(f"    grid  = gt['grid_xy']      # (10000,2) — punkty ewaluacji")
    print(f"    mload = gt['mask_load']    # (10000,)  bool — strefa obciążenia")
    print(f"    with torch.no_grad():")
    print(f"        uv = model(torch.tensor(grid, dtype=torch.float32)).numpy()")
    print(f"    l2_u = np.sqrt(np.mean((uv[:,0] - u_gt)**2))")
    print(f"{'='*65}\n")

    return out_path


if __name__ == "__main__":
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else nx
    run_highres_fem(nx=nx, ny=ny)
