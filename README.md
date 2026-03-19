# Projekt PINN: Liniowa Sprężystość 2D

Instrukcja przygotowania środowiska i uruchomienia notatników po sklonowaniu repozytorium.

## 0. Wymagania wstępne
Przed przystąpieniem do pracy upewnij się, że w systemie zainstalowane są:
- **Python** (zalecana wersja 3.8 - 3.11) wraz z menedżerem pakietów `pip`.
- *(Opcjonalnie, dla obliczeń na karcie graficznej)*: Zainstalowane sterowniki NVIDIA oraz odpowiednia wersja środowiska PyTorch wspierająca **CUDA** (w takim przypadku instalacja PyTorch może wymagać innej komendy instalacyjnej, zgodnej z instrukcją na stronie [pytorch.org](https://pytorch.org/get-started/locally/)).

## 1. Utworzenie wirtualnego środowiska
W głównym folderze projektu uruchom w terminalu poniższą komendę, aby utworzyć izolowane środowisko o nazwie `venv`:
```bash
python -m venv venv
```

## 2. Aktywacja wirtualnego środowiska
Następnie należy je aktywować w zależności od używanego systemu operacyjnego:

**Linux / macOS:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt / PowerShell):**
```cmd
venv\Scripts\activate
```

## 3. Instalacja pakietów
Mając aktywne środowisko (z prefixem `(venv)` w konsoli), zainstaluj wszystkie niezbędne biblioteki korzystając z przygotowanego pliku listującego pakiety:
```bash
pip install -r requirements.txt
```

## 4. Uruchomienie notatników interaktywnych (Opcja 1)
W celu interaktywnej pracy z kodem i wizualizacjami, uruchom środowisko graficzne Jupyter (ja polecam VS Code z wtyczkami Jupyter):
```bash
jupyter notebook
```
Następnie w przeglądarce internetowej przejdź do katalogu `pinn-notebooks` i otwórz wybrany plik, na przykład `PINN_elasticity_2d_baseline.ipynb`.

## 5. Wykonanie skryptu w terminalu (Opcja 2)
W przypadku chęci przeprowadzenia obliczeń w tle, bez konieczności uruchamiania środowiska Jupyter, możliwe jest wykonanie gotowego skryptu w języku Python. 

Proces nie blokuje terminala oknami graficznymi i automatycznie generuje wykres przebiegu uczenia. Aby go uruchomić, wykonaj z poziomu głównego katalogu polecenie:
```bash
python pinn-notebooks/PINN_elasticity_2d_baseline.py
```
Wyniki eksperymentu w formacie `.npz` oraz wygenerowany wykres (np. `baseline_plot.png`) zostaną zapisane w podkatalogu `pinn-notebooks/results/`.
