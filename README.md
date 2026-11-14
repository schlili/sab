# Hangrobbanás atomi szintű vizsgálata klasszikus mechanikai modellen (`sab`)

Ez a projekt egy numerikus fizikai szimuláció (`sab.py`), amely egy $R$ sugarú körpályán elhelyezkedő, harmonikusan csatolt (rugókkal összekötött) $n$ darab tömegpontból álló kör, valamint a gyűrű körül $\rho$ sugarú külső körpályán mozgó perturbáló golyó csatolt dinamikáját vizsgálja.

## Fizikai modell

A rendszer két fő részből épül fel:

- **Belső kör ($n$ darab golyó):** $R$ sugarú körön mozgó, $m$ tömegű részecskék, amelyeket $d$ rugóállandójú, $l_0 = 2R \sin(\pi/n)$ egyensúlyi hosszúságú rugók kapcsolnak össze. Helyzetüket a $\phi_i(t)$ szögek, szögsebességüket az $\omega_i(t)$ változók írják le.
- **Külső golyó:** $\rho$ sugarú körpályán mozgó részecske ($\theta(t)$ szög, $\Omega(t)$ szögsebesség), amely exponenciálisan lecsengő kölcsönhatásban áll a belső kör összes elemével:

$$
V_{\mathrm{out}} = -k \sum_{i=1}^{N} \exp(-b r_i)
$$

ahol $r_i = \sqrt{R^2 - 2R\rho\cos(\theta - \phi_i) + \rho^2}$ a külső és az $i$-edik belső tömegpont közötti távolság.

## Követelmények

A szimuláció futtatásához Python 3 és az alábbi csomagok szükségesek (Jupyter Notebook / IPython környezetben a beágyazott animációk megjelenítéséhez):

```bash
pip install numpy scipy matplotlib ipython tqdm
```

## A `Modell` osztály felépítése (`sab.py`)

### Paraméterek

| Paraméter | Jelentés |
| :--- | :--- |
| `n` | A belső kört alkotó golyók száma. |
| `m` | A golyók tömege. |
| `d` | A belső golyókat összekötő rugók rugóállandója. |
| `R` | A belső kör sugara. |
| `rho` ($\rho$) | A külső golyó pályájának sugara. |
| `k` | A külső és belső golyók közötti kölcsönhatás erőssége. |
| `b` | A kölcsönhatás térbeli lecsengési együtthatója. |
| `o` ($\Omega_0$) | A külső golyó kezdeti szögsebessége. |
| `u` | A mozgásegyenletek megoldása. |
| `t` | A kiértékelt időpontok. |

### Főbb metódusok

- `Modell(file=None)` – Új szimulációs objektum létrehozása, vagy korábbi állapot betöltése `.pkl` fájlból.
- `change_params(params)` – A fizikai paraméterek beállítása egy `SimpleNamespace` objektumon keresztül.
- `start_maker(n)` – Kezdeti feltételek generálása: egyenletes szögbeli eloszlás a belső gyűrűn ($\omega_0 = 0$) és $\Omega_0$ kezdeti szögsebesség a külső részecskén.
- `solve_ivp(t, rtol=1e-3, atol=1e-6, max_step=0.01)` – A mozgásegyenletek numerikus integrálása Runge–Kutta (`RK45`) módszerrel.
- `solution_divider(t)` – A folytonos megoldás kiértékelése adott $t$ időrácsra, valamint a $\phi$, $\omega$, $\theta$, $\Omega$ tömbök szétválasztása.
- `energy_calc()` – A rendszer kinetikus ($K_{\mathrm{in}}$, $K_{\mathrm{out}}$), potenciális ($V_{\mathrm{in}}$, $V_{\mathrm{out}}$) és teljes energiájának ($E$) kiszámítása az energiamegmaradás ellenőrzéséhez.
- `phonon_spectrum(ax, cmap)` – 2D gyors Fourier-transzformáció (FFT) segítségével kiszámítja és ábrázolja a belső gyűrű diszperziós relációját ($\omega(k)$ fononspektrum).
- `animation(title, zoom=True)` – Interaktív HTML5 animáció készítése a részecskék mozgásáról (opcionálisan a külső részecske koordináta-rendszerére nagyítva).
- `save(file)` – A teljes modellállapot és a megoldás mentése `.pkl` fájlba.

## Példa a használatra

```python
import numpy as np
from scipy.integrate import *
from numpy.fft import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from IPython.display import HTML, display
from matplotlib.ticker import ScalarFormatter
from types import SimpleNamespace
import pickle
from tqdm.auto import tqdm
import sab

# 1. Modell példányosítása és paraméterek megadása
sim = sab.Modell()
params = SimpleNamespace(
    d=1.0, m=1.0, R=3.0, rho=3.1,
    k=0.5, b=60.0, o=0.03, u=np.array([]), t=np.array([])
)
sim.change_params(params)

# 2. Kezdeti feltételek és numerikus integrálás
sim.start_maker(n=300)
sim.solve_ivp(t=120)

# 3. Eredmények kiértékelése adott időrácsra
t_grid = np.linspace(0, 120, 120 * 100)
sim.solution_divider(t_grid)

# 4. Energiamegmaradás és fononspektrum vizsgálata
sim.energy_calc()
fig, ax = plt.subplots()
sim.phonon_spectrum(ax, cmap='viridis')
plt.show()

# 5. Eredmények mentése
sim.save("szimulacio_eredmeny")
```

## Szerző

- Schmidt Liliána Éva ([@schlili](https://github.com/schlili))
