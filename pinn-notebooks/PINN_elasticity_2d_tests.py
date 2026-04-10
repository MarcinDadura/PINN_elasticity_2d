import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Importujemy klasy i funkcje z Twoich plików
import PINN_elasticity_2d_baseline as base
import PINN_elasticity_2d_param as param

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Platforma testowa uruchomiona na: {device}")

# ── 1. Funkcje Trenujące z pomiarem czasu ─────────────────────────────────────

def train_and_save_baseline(e_target, epochs):
    print(f"\n--- Trening BASELINE dla E = {e_target} GPa ---")
    pinn = base.PINN(base.LAYERS, base.NEURONS_PER_LAYER, act=torch.nn.Tanh()).to(device)
    
    loss_fn = base.Loss(
        x_domain=(0.0, base.LENGTH),
        y_domain=(0.0, base.LENGTH),
        n_points=base.N_POINTS,
        n_bnd_points=base.N_BND_POINTS,
        device=device,
        weight_r=base.WEIGHT_RESIDUAL,
        weight_b=base.WEIGHT_BOUNDARY,
        E=e_target, 
        nu=base.NU
    )

    start_time = time.time()
    trained_pinn, histories = base.train_model(pinn, loss_fn, learning_rate=0.002, max_epochs=epochs, log_interval=500)
    train_time = time.time() - start_time
    
    os.makedirs("models", exist_ok=True)
    torch.save(trained_pinn.state_dict(), f"models/baseline_e{e_target}.pth")
    np.savez(f"models/baseline_e{e_target}_hist.npz", **histories)
    print("Zapisano model bazowy.")
    
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
    plot_path = os.path.join(f"models", f"baseline_plot_{e_target}.png")
    plt.savefig(plot_path)
    print(f"Wykres przebiegu nauki zapisano do: {plot_path}")
    return trained_pinn, train_time

def train_and_save_param(epochs):
    print("\n--- Trening PARAMETRYCZNEGO (E = 70.0 110.0 200.0 GPa) ---")
    pinn = param.PINN(param.LAYERS, param.NEURONS_PER_LAYER, act=torch.nn.Tanh()).to(device)
    
    # Zakres nadpisany na potrzeby ujednolicenia z testem, jeśli nie zmieniono w pliku
    loss_fn = param.Loss(
        x_domain=(0.0, param.LENGTH),
        y_domain=(0.0, param.LENGTH),
        e_values=[70.0, 110.0, 200.0],
        n_points=param.N_POINTS,
        n_bnd_points=param.N_BND_POINTS,
        device=device,
        weight_r=param.WEIGHT_RESIDUAL,
        weight_b=param.WEIGHT_BOUNDARY,
        nu=param.NU
    )

    start_time = time.time()
    trained_pinn, histories = param.train_model(pinn, loss_fn, learning_rate=0.002, max_epochs=epochs, log_interval=500)
    train_time = time.time() - start_time
    
    os.makedirs("models", exist_ok=True)
    torch.save(trained_pinn.state_dict(), "models/param_model.pth")
    np.savez("models/param_model_hist.npz", **histories)
    print("Zapisano model parametryczny.")

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
    plot_path = os.path.join(f"models", "param_plot.png")
    plt.savefig(plot_path)
    print(f"Wykres przebiegu nauki zapisano do: {plot_path}")
    return trained_pinn, train_time

# ── 3. Ewaluacja, logowanie i wykresy ─────────────────────────────────────────

def compute_stats(u_b, v_b, u_p, v_p):
    diff_u = np.abs(u_b - u_p)
    diff_v = np.abs(v_b - v_p)
    return {
        "U - Maksymalna różnica (Max Error)": np.max(diff_u),
        "U - Średni błąd bezwzględny (MAE)": np.mean(diff_u),
        "U - Błąd średniokwadratowy (MSE)": np.mean(diff_u**2),
        "U - Odchylenie standardowe różnicy": np.std(diff_u),
        "V - Maksymalna różnica (Max Error)": np.max(diff_v),
        "V - Średni błąd bezwzględny (MAE)": np.mean(diff_v),
        "V - Błąd średniokwadratowy (MSE)": np.mean(diff_v**2),
        "V - Odchylenie standardowe różnicy": np.std(diff_v),
    }

# --- Funkcja do generowania wykresów (wyciągnięta dla reużywalności) ---
def save_comparison_plot(u_b, v_b, u_p, v_p, e_test, suffix=""):
    u_min = min(np.min(u_b), np.min(u_p))
    u_max = max(np.max(u_b), np.max(u_p))
    v_min = min(np.min(v_b), np.min(v_b))
    v_max = max(np.max(v_b), np.max(v_p))

    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    fig.suptitle(f'Porównanie modeli dla materiałów o E = {e_test}', fontsize=20)

    # Parametry dla osi U (wspólne vmin i vmax)
    u_kwargs = {'extent': [0, 1, 0, 1], 'origin': 'lower', 'cmap': 'viridis', 'vmin': u_min, 'vmax': u_max}
    
    # Wiersz 1: U
    im0 = axes[0, 0].imshow(u_b, **u_kwargs)
    axes[0, 0].set_title(f"Baseline: Przemieszczenie U {suffix}")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(u_p, **u_kwargs)
    axes[0, 1].set_title(f"Parametric: Przemieszczenie U {suffix}")
    fig.colorbar(im1, ax=axes[0, 1])

    # Parametry dla osi V (wspólne vmin i vmax)
    v_kwargs = {'extent': [0, 1, 0, 1], 'origin': 'lower', 'cmap': 'viridis', 'vmin': v_min, 'vmax': v_max}

    # Wiersz 2: V
    im2 = axes[1, 0].imshow(v_b, **v_kwargs)
    axes[1, 0].set_title(f"Baseline: Przemieszczenie V {suffix}")
    fig.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(v_p, **v_kwargs)
    axes[1, 1].set_title(f"Parametric: Przemieszczenie V {suffix}")
    fig.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f"models/comparison_E{e_test}_{suffix}.png", dpi=150)
    plt.close() # Zamknięcie figury zapobiega nakładaniu się wykresów w pamięci
    print(f"Wykresy zapisano jako models/comparison_E{e_test}_{suffix}.png")

def evaluate_and_compare(model_b, model_p, e_test, report_file, iter, resolution=1000):
    print(f"\n[{e_test}] --- Rozpoczynam ewaluację na siatce {resolution}x{resolution} dla E = {e_test} ---")
    model_b.eval()
    model_p.eval()

    x_lin = torch.linspace(0.0, 1.0, resolution, device=device)
    y_lin = torch.linspace(0.0, 1.0, resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x_lin, y_lin, indexing="ij")
    
    flat_x = grid_x.reshape(-1, 1).requires_grad_(True)
    flat_y = grid_y.reshape(-1, 1).requires_grad_(True)
    flat_e = torch.full_like(flat_x, e_test)

    # Obliczenia w trybie no_grad() zapobiegają błędom braku pamięci (OOM) przy milionie punktów
    t_start = time.perf_counter()
    with torch.no_state_grad() if hasattr(torch, "no_state_grad") else torch.no_grad():
        u_b, v_b = model_b(flat_x, flat_y)
        u_p, v_p = model_p(flat_x, flat_y, flat_e)
    t_end = time.perf_counter()
    print(f"Czas obliczeń dla 1 000 000 punktów: {t_end - t_start:.3f} s")

    u_b_np = u_b.reshape(resolution, resolution).cpu().numpy()
    v_b_np = v_b.reshape(resolution, resolution).cpu().numpy()

    u_p_np = u_p.reshape(resolution, resolution).cpu().numpy()
    v_p_np = v_p.reshape(resolution, resolution).cpu().numpy()

    stats = compute_stats(u_b_np, v_b_np, u_p_np, v_p_np)
    # --- Krok C: Statystyki ---

    print("\nSTATYSTYKI PORÓWNAWCZE (Baseline vs Parametric):")
    for key, value in stats.items():
        print(f" * {key}: {value:.6e}")
    with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- Iteracja {iter} (E={e_test}) ---\n")
            for key, value in stats.items():
                f.write(f" * {key}: {value:.6e}\n")
    
    return u_b_np, v_b_np, u_p_np, v_p_np

def export_to_vtk(filename, u, v, resolution, length=1.0):
    """
    Zapisuje pola przemieszczeń do pliku ASCII VTK.
    Dane są zapisywane jako pole wektorowe (U, V, 0.0), co ułatwia użycie 
    filtru 'Warp By Vector' w programie ParaView.
    """
    points_count = resolution * resolution
    spacing = length / (resolution - 1)
    
    # Siatka grid_x, grid_y była generowana z indexing="ij", 
    # więc macierze u i v mają kształt [x_idx, y_idx].
    # Format VTK STRUCTURED_POINTS wymaga, by współrzędna X zmieniała się najszybciej.
    # Transpozycja (.T) zmienia układ na [y_idx, x_idx], a .flatten() spłaszcza to
    # iterując od najgłębszego poziomu (czyli najszybciej idzie po X).
    u_flat = u.T.flatten()
    v_flat = v.T.flatten()
    
    with open(filename, 'w', encoding='utf-8') as f:
        # Nagłówek VTK
        f.write("# vtk DataFile Version 3.0\n")
        f.write("PINN 2D Elasticity Displacement Vector Field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        
        # Wymiary siatki (Z = 1 dla 2D)
        f.write(f"DIMENSIONS {resolution} {resolution} 1\n")
        f.write("ORIGIN 0.0 0.0 0.0\n")
        f.write(f"SPACING {spacing:.6f} {spacing:.6f} 1.0\n")
        
        # Zapisanie danych punktowych (wektorowych)
        f.write(f"POINT_DATA {points_count}\n")
        f.write("VECTORS displacement float\n")
        
        # Wypisanie wartości: X Y Z (gdzie Z w 2D to 0.0)
        for i in range(points_count):
            f.write(f"{u_flat[i]:.6e} {v_flat[i]:.6e} 0.0\n")

# ── 4. Uruchomienie Główne ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Materiały: Aluminium (70), Tytan/Miedź (110), Stal (200)
    MATERIALS_E = [70.0, 110.0, 170.0, 200.0]
    EPOCHS_TO_TRAIN = 30000
    REPORT_FILE = "models/training_report.txt"

    os.makedirs("models", exist_ok=True)
    
    # Inicjalizacja raportu
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== RAPORT Z TRENINGU I TESTÓW SIECI PINN ===\n\n")
        f.write("CZASY TRENINGU:\n")

    # 1. Trenowanie modelu parametrycznego (jeden model dla wszystkich E)
    model_param = param.PINN(param.LAYERS, param.NEURONS_PER_LAYER).to(device)
    param_path = "models/param_model.pth"
    if os.path.exists(param_path):
        print("\n[INFO] Wczytano gotowy model parametryczny.")
        model_param.load_state_dict(torch.load(param_path, map_location=device))
    else:
        model_param, time_p = train_and_save_param(EPOCHS_TO_TRAIN)
        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(f"  - Model Parametryczny: {time_p:.2f} s\n")

    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write("\nSTATYSTYKI BŁĘDU (EWALUACJA):\n")

    # 2. Pętla przez materiały: trening dedykowanych baseline i porównanie
    for e_val in MATERIALS_E:
        model_baseline = base.PINN(base.LAYERS, base.NEURONS_PER_LAYER).to(device)
        baseline_path = f"models/baseline_e{e_val}.pth"
        
        if os.path.exists(baseline_path):
            print(f"\n[INFO] Wczytano gotowy model baseline dla E={e_val} GPa.")
            model_baseline.load_state_dict(torch.load(baseline_path, map_location=device))
        else:
            model_baseline, time_b = train_and_save_baseline(e_val, EPOCHS_TO_TRAIN)
            with open(REPORT_FILE, "a", encoding="utf-8") as f:
                f.write(f"  - Model Baseline (E={e_val}): {time_b:.2f} s\n")
        u_p_list, v_p_list = [], []
        u_b_list, v_b_list = [], []
        # Uruchomienie porównania
        for i in range(5):
            ub, vb, up, vp = evaluate_and_compare(model_baseline, model_param, e_test=e_val, iter=i, report_file=REPORT_FILE)
            u_b_list.append(ub)
            v_b_list.append(vb)
            u_p_list.append(up)
            v_p_list.append(vp)
            # 1. Obliczanie średniej z 5 iteracji
        u_b_avg = np.mean(u_b_list, axis=0)
        v_b_avg = np.mean(v_b_list, axis=0)
        u_p_avg = np.mean(u_p_list, axis=0)
        v_p_avg = np.mean(v_p_list, axis=0)

        # 2. Statystyki dla wartości średnich
        avg_stats = compute_stats(u_b_avg, v_b_avg, u_p_avg, v_p_avg)
        
        print(f"\nSTATYSTYKI DLA ŚREDNIEJ (E={e_val}):")
        for k, v in avg_stats.items():
            print(f" [AVG] {k}: {v:.6e}")

        # 3. Zapis do pliku
        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n=== WYNIKI ŚREDNIE (z 5 prób) dla E = {e_val} ===\n")
            for key, value in avg_stats.items():
                f.write(f" * Średni {key}: {value:.6e}\n")
            f.write("-" * 40 + "\n")

        # 4. Wykres dla średnich
        save_comparison_plot(u_b_avg, v_b_avg, u_p_avg, v_p_avg, e_val, suffix="_AVERAGE")

        # 5. Eksport średnich wyników do ParaView (VTK)
        vtk_base_file = f"models/displacement_baseline_E{e_val}.vtk"
        vtk_param_file = f"models/displacement_param_E{e_val}.vtk"
        
        export_to_vtk(vtk_base_file, u_b_avg, v_b_avg, resolution=1000, length=1.0)
        export_to_vtk(vtk_param_file, u_p_avg, v_p_avg, resolution=1000, length=1.0)
        print(f" [INFO] Zapisano pliki VTK dla ParaView: {vtk_base_file}, {vtk_param_file}")
    
    print(f"\n[SUKCES] Pełny raport został zapisany w pliku: {REPORT_FILE}")