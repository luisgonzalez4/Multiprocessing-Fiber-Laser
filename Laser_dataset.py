import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import TeSpeS as ts
import pulse as ps
import fibre as fb


def semilla(n, nt, t, t0, p0):
    A = np.exp(-0.5 * (t / t0)**2) * np.random.uniform(0, p0, (n, nt))
    phase = np.random.uniform(0, 2*np.pi, (n, nt))
    A = A * np.exp(1j * phase)
    return A


def laser(args):
    idx, A, RT, t, omega, lambda0 = args

    for i in range(RT):
        A = fb.DopedFibre(A, t, omega, lambda0, 10e-9, 4e-26, 6e-3, 0.73, 1e-9, 10, 10*500)
        A = fb.SMF(A, omega, -2.17e-26, 8.6e-41, 1.1e-3, 7, 7*500)
        Aout, A = fb.Coupler(A, 0.1)
        A = fb.SA(A, 0.9, 150)
        A = ts.filter_gaussian(A, omega, lambda0, 10e-9, 2)
        A = fb.SMF(A, omega, -2.17e-26, 8.6e-41, 1.1e-3, 8, 8*500)
        print(f"Pulse {idx+1}, Round {i+1}/{RT}",end="\r",flush=True)
    return idx, Aout


def multiprocess(Ain, workers, RT, t, omega, lambda0):
    t_start = time.time()
    total = len(Ain)

    tareas = [(i, Ain[i], RT, t, omega, lambda0) for i in range(total)]
    resultados = [None] * total

    with mp.Pool(processes=workers) as pool:
        for k, (idx, res) in enumerate(pool.imap_unordered(laser, tareas), start=1):
            resultados[idx] = res
            avance = 100 * k / total
            avance = k / total
            porcentaje = avance * 100
            longitud = 30
            llenos = int(longitud * avance)
            barra = "█" * llenos + "░" * (longitud - llenos)
            print(f"\r{barra} {porcentaje:5.1f}% ({k}/{total})", end="", flush=True)

    return time.time() - t_start, np.array(resultados)


if __name__ == "__main__":
    nt = 2**13
    Tmax = 60e-12
    lambda0 = 1550e-9
    dt, t, omega, lambda_nm = ts.window(nt, Tmax, lambda0)

    t0 = 1e-12
    n = 20
    P0 = 150
    RT = 2

    A = semilla(n, nt, t, t0, P0)

    workers = min(mp.cpu_count(), n)
    print(f"Usando {workers} núcleos para: {n} pulsos.")
    t_par, Aout = multiprocess(A, workers, RT, t, omega, lambda0)

    print("\nTiempo de simulación:", t_par/3600)
    print("Shape de Aout:", Aout.shape)

    Iout = ps.Intensity(Aout)
    Spec = ps.Spectrum(Aout)
    print('Guardando...')
    pd.DataFrame(Iout).to_csv("/Users/luisgonzalez/Desktop/FLML/Aout_intensity.csv", index=False, header=False)
    pd.DataFrame(Spec).to_csv("/Users/luisgonzalez/Desktop/FLML/Aout_spectrum.csv", index=False, header=False)

    print('Archivos guardados: Aout_intensity.csv y Aout_spectrum.csv')

    plt.plot(t*1e12, Iout[0])
    plt.plot(t*1e12, Iout[1])
    plt.plot(t*1e12, Iout[-1])
    plt.xlabel("Tiempo (ps)")
    plt.ylabel("Intensidad (a.u.)")
    plt.show()