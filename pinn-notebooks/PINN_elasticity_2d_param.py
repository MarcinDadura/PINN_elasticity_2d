import os
import time
import tracemalloc
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie obliczeniowe: {device}")

# ── 1. Parametry ──────────────────────────────────────────────────────────────
#sprawdzic jednostki!, model bazowy, nie musi miec sensu, wziąć materiały w stylu gąbka (sprężysty), stal (200 GPa), tytan/miedz (110GPa), aluminium (70GPa) - do sprawdzenia
LENGTH = 1.0                 # 1m x 1m [m] Wymiar płyty
NU = 0.3                     # [-] Współczynnik Poissona
E_VALUES = [70.0, 110.0, 200.0] # [GPa] Aluminium, Tytan/Miedź, Stal
N_POINTS = 30                # Punkty wewnątrz domeny
N_BND_POINTS = 30            # Punkty na krawędziach
WEIGHT_RESIDUAL = 1.0        # Waga błędu fizycznego (PDE)
WEIGHT_BOUNDARY = 10.0       # Waga błędu warunków brzegowych
LAYERS = 4                   
NEURONS_PER_LAYER = 80       
EPOCHS = 30000               
LEARNING_RATE = 0.002        
LOG_INTERVAL = 500           
PINN_INIT_SEED = 42          

# ── 2. Architektura Sieci ─────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):
        super().__init__()
        self.layer_in = nn.Linear(3, dim_hidden) # Wejścia: x, y, E
        # siec sie moze lepiej uczyc jak bedziemy zwrac tez pochodne po przemieszczeniu - warto dodac wiecej wyjsc
        self.layer_out = nn.Linear(dim_hidden, 2) # Wyjścia: u, v
        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x, y, e):
        # Łączymy x, y oraz e w jeden tensor wejściowy
        x_stack = torch.cat([x, y, e], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)
        return logits[:, 0:1], logits[:, 1:2]

    def device(self):
        return next(self.parameters()).device

def f_eval(pinn: PINN, x: torch.Tensor, y: torch.Tensor, e: torch.Tensor):
    return pinn(x, y, e)

def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(df_value),
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_value

# ── 3. Generatory Punktów (Dla dyskretnych wartości E) ────────────────────────
def get_interior_points(x_domain, y_domain, e_values, n_points, device):
    x_lin = torch.linspace(x_domain[0], x_domain[1], steps=n_points, device=device)
    y_lin = torch.linspace(y_domain[0], y_domain[1], steps=n_points, device=device)
    #Tworzymy tensor bezpośrednio z podanej listy
    e_lin = torch.tensor(e_values, dtype=torch.float32, device=device)
    
    grids = torch.meshgrid(x_lin, y_lin, e_lin, indexing="ij")
    x = grids[0].reshape(-1, 1).requires_grad_(True)
    y = grids[1].reshape(-1, 1).requires_grad_(True)
    e = grids[2].reshape(-1, 1).requires_grad_(True)
    return x, y, e

def get_boundary_points(x_domain, y_domain, e_values, n_points, device):
    x_lin = torch.linspace(x_domain[0], x_domain[1], steps=n_points, device=device)
    y_lin = torch.linspace(y_domain[0], y_domain[1], steps=n_points, device=device)
    e_lin = torch.tensor(e_values, dtype=torch.float32, device=device)
    
    def augment_with_e(x_bnd, y_bnd):
        x_grid, e_grid = torch.meshgrid(x_bnd, e_lin, indexing="ij")
        y_grid, _ = torch.meshgrid(y_bnd, e_lin, indexing="ij")
        return x_grid.reshape(-1, 1).requires_grad_(True), \
               y_grid.reshape(-1, 1).requires_grad_(True), \
               e_grid.reshape(-1, 1).requires_grad_(True)

    b_l = augment_with_e(torch.full_like(y_lin, x_domain[0]), y_lin)
    b_r = augment_with_e(torch.full_like(y_lin, x_domain[1]), y_lin)
    b_b = augment_with_e(x_lin, torch.full_like(x_lin, y_domain[0]))
    b_t = augment_with_e(x_lin, torch.full_like(x_lin, y_domain[1]))
    
    return b_l, b_r, b_b, b_t

def get_specific_points(coord_x, coord_y, e_values, device):
    e_lin = torch.tensor(e_values, dtype=torch.float32, device=device)
    e_points = len(e_values)
    x = torch.full((e_points, 1), coord_x, device=device, requires_grad=True)
    y = torch.full((e_points, 1), coord_y, device=device, requires_grad=True)
    e = e_lin.reshape(-1, 1).requires_grad_(True)
    return x, y, e

# ── 4. Funkcja Straty ─────────────────────────────────────────────────────────
class Loss:
    #Zamiast e_domain i n_e przyjmujemy e_values (listę)
    def __init__(self, x_domain, y_domain, e_values, n_points, n_bnd_points, device, weight_r=1.0, weight_b=1.0, nu=0.3):
        self.weight_r = weight_r
        self.weight_b = weight_b
        self.nu = nu

        self.interior_points = get_interior_points(x_domain, y_domain, e_values, n_points, device)
        self.boundary_points = get_boundary_points(x_domain, y_domain, e_values, n_bnd_points, device)
        self.center_d1 = get_specific_points(0.5, 0.5, e_values, device)
        self.corner_d2 = get_specific_points(1.0, 0.0, e_values, device)

#przygotowac rownania i rysunki na prezentacje, dodac sily
    def get_stresses(self, pinn, x, y, e):
        # Parametry zależne od tensora E
        lmbda = (e * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = e / (2 * (1 + self.nu))
        lmbda_ps = (2 * mu * lmbda) / (lmbda + 2 * mu)

        u, v = f_eval(pinn, x, y, e)
        u_x = df(u, x, order=1)
        u_y = df(u, y, order=1)
        v_x = df(v, x, order=1)
        v_y = df(v, y, order=1)

        eps_xx, eps_yy, eps_xy = u_x, v_y, 0.5 * (u_y + v_x)

        sigma_xx = (lmbda_ps + 2 * mu) * eps_xx + lmbda_ps * eps_yy
        sigma_yy = (lmbda_ps + 2 * mu) * eps_yy + lmbda_ps * eps_xx
        sigma_xy = 2 * mu * eps_xy

        return sigma_xx, sigma_yy, sigma_xy

    def residual_loss(self, pinn):
        x, y, e = self.interior_points
        sigma_xx, sigma_yy, sigma_xy = self.get_stresses(pinn, x, y, e)
        eq_x = df(sigma_xx, x, order=1) + df(sigma_xy, y, order=1)
        eq_y = df(sigma_xy, x, order=1) + df(sigma_yy, y, order=1)
        return (eq_x**2).mean() + (eq_y**2).mean()

    def boundary_loss(self, pinn):
        # Dirichlet D1: u=0, v=0 w centrum dla wszystkich E
        xc, yc, ec = self.center_d1
        uc, vc = f_eval(pinn, xc, yc, ec)
        loss_d1 = (uc**2 + vc**2).mean()

        # Dirichlet D2: v=0 w rogu dla wszystkich E
        xd2, yd2, ed2 = self.corner_d2
        _, vd2 = f_eval(pinn, xd2, yd2, ed2)
        loss_d2 = (vd2**2).mean()

        (b_l, b_r, b_b, b_t) = self.boundary_points

        # Lewa krawędź (Neumann: free)
        s_xx_l, _, s_xy_l = self.get_stresses(pinn, *b_l)
        loss_left = (s_xx_l**2 + s_xy_l**2).mean()

        # Dolna i Górna (Neumann: free)
        _, s_yy_b, s_xy_b = self.get_stresses(pinn, *b_b)
        _, s_yy_t, s_xy_t = self.get_stresses(pinn, *b_t)
        loss_bottom = (s_yy_b**2 + s_xy_b**2).mean()
        loss_top = (s_yy_t**2 + s_xy_t**2).mean()

        # Prawa krawędź (Obciążenie) - sprawdzic jak dziala sila
        xr, yr, er = b_r
        s_xx_r, _, s_xy_r = self.get_stresses(pinn, xr, yr, er)
        force_mask = (yr >= 0.9).float()
        loss_right = ((s_xx_r - force_mask)**2 + s_xy_r**2).mean()

        return loss_d1 + loss_d2 + loss_left + loss_bottom + loss_top + loss_right

    def verbose(self, pinn):
        res_loss = self.residual_loss(pinn)
        bnd_loss = self.boundary_loss(pinn)
        total = self.weight_r * res_loss + self.weight_b * bnd_loss
        return total, res_loss, bnd_loss

    def __call__(self, pinn):
        return self.verbose(pinn)[0]

# ── 5. Pętla Trenująca ────────────────────────────────────────────────────────
def train_model(nn_approximator, loss_fn, learning_rate=0.002, max_epochs=30000, log_interval=500):
    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    loss_history = []
    pde_history  = []
    bc_history   = []
    history_epochs = []

    for epoch in range(max_epochs):
        try:
            loss = loss_fn(nn_approximator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if (epoch + 1) % log_interval == 0:
                l_tot, l_res, l_bc = loss_fn.verbose(nn_approximator)
                pde_history.append(l_res.item())
                bc_history.append(l_bc.item())
                history_epochs.append(epoch + 1)
                print(
                    f"Ep {epoch+1:6d} | Tot: {l_tot.item():.4e} "
                    f"| PDE: {l_res.item():.4e} | BC: {l_bc.item():.4e}"
                )

        except KeyboardInterrupt:
            print("Trening przerwany przez użytkownika.")
            break

    histories = {
        "loss_history":   np.array(loss_history),
        "pde_history":    np.array(pde_history),
        "bc_history":     np.array(bc_history),
        "history_epochs": np.array(history_epochs),
    }
    return nn_approximator, histories

# ── 6. Główna funkcja wykonawcza ──────────────────────────────────────────────
def main():
    torch.manual_seed(PINN_INIT_SEED)
    np.random.seed(PINN_INIT_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(PINN_INIT_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    pinn = PINN(LAYERS, NEURONS_PER_LAYER, act=nn.Tanh()).to(device)
    
    # Przygotowanie domen dla tensorów
    x_domain = (0.0, LENGTH)
    y_domain = (0.0, LENGTH)

    loss_fn = Loss(x_domain, y_domain, E_VALUES, N_POINTS, N_BND_POINTS, device, WEIGHT_RESIDUAL, WEIGHT_BOUNDARY, NU)

    print(f"Rozpoczynam trening modelu dla dyskretnych wartości E: {E_VALUES} GPa...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tracemalloc.start()
    t_start = time.perf_counter()
    trained_pinn, histories = train_model(pinn, loss_fn, LEARNING_RATE, EPOCHS, LOG_INTERVAL)

    t_end = time.perf_counter()
    _, peak_cpu_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    training_time   = t_end - t_start
    peak_cpu_mb     = peak_cpu_bytes / 1024**2
    peak_gpu_mb     = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0

    final_total, final_pde, final_bc = loss_fn.verbose(trained_pinn)

    print(f"\nCzas treningu      : {training_time:.1f} s")
    print(f"RAM szczytowo (CPU): {peak_cpu_mb:.1f} MB")
    print(f"RAM szczytowo (GPU): {peak_gpu_mb:.1f} MB")
    print(f"Końcowy PDE Res    : {final_pde.item():.4e}")
    print(f"Końcowy BC Loss    : {final_bc.item():.4e}")

    # Zapis wyników
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "baseline_results.npz")

    np.savez(
        save_path,
        loss_history    = histories["loss_history"],
        pde_history     = histories["pde_history"],
        bc_history      = histories["bc_history"],
        history_epochs  = histories["history_epochs"],
        final_pde       = np.array(final_pde.item()),
        final_bc        = np.array(final_bc.item()),
        training_time_s = np.array(training_time),
        peak_cpu_mb     = np.array(peak_cpu_mb),
        peak_gpu_mb     = np.array(peak_gpu_mb),
        n_collocation_points = np.array(N_POINTS ** 2),
        method_name          = np.array("Baseline (Uniform Grid, bc-fixed, 30k)"),
        pinn_init_seed       = np.array(PINN_INIT_SEED),
        epochs               = np.array(EPOCHS),
        lr                   = np.array(LEARNING_RATE),
        lr_schedule          = np.array("fixed_0.002"),
    )
    print(f"Wyniki zapisane do: {save_path}")

    # Wizualizacja i zapis wykresu
    def running_average(y, window=50):
        cumsum = np.cumsum(np.insert(y, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / float(window)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    axes[0].plot(running_average(histories["loss_history"]), color="blue", label="Total Loss (wygładzony)")
    axes[0].set_title("Baseline: Total Loss (Uniform Grid)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Wartość Loss")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)
    axes[0].legend()

    ep = histories["history_epochs"]
    axes[1].plot(ep, histories["pde_history"], color="red",   label="PDE Residual")
    axes[1].plot(ep, histories["bc_history"],  color="green", label="BC Loss")
    axes[1].set_title("Baseline: PDE vs BC Loss")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Wartość Loss")
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "baseline_plot.png")
    plt.savefig(plot_path)
    print(f"Wykres przebiegu nauki zapisano do: {plot_path}")

if __name__ == "__main__":
    main()
