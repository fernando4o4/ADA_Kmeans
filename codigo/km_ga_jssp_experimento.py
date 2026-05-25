from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# ==============================================================================
# KM-GA PARA JSSP - EXPERIMENTO FINAL A-D EN TRES CASOS
# ==============================================================================
# Base reutilizada:
# - cruza_pmx() y calcular_makespan() vienen del archivo Python T8/T9 entregado.
# - La estructura del AG conserva torneo, PMX, mutacion dinamica y elitismo.
#
# Integracion implementada:
# - Configuracion A: poblacion 100% aleatoria.
# - Configuracion B: 70% K-Means Perfil 1 + SPT, 30% aleatoria.
# - Configuracion C: Perfil 1 + SPT, Perfil 2 + SPT y aleatoria.
# - Configuracion D: mezcla de Perfil 1/2 con SPT, LPT, MWR y MOPNR.
# - Casos evaluados: Base, Mediano y Grande.
# - Ejecucion con semillas 1, 2 y 3.
# - Guardado de resultados y convergencia en CSV.
# ==============================================================================


N_INDIVIDUOS = 20
M_GENERACIONES = 50
K_TORNEO = 3
K_CLUSTERS = 3
SEMILLAS = [1, 2, 3]
CONFIGURACIONES = ["A", "B", "C", "D"]

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "resultados_km_ga_jssp_tres_casos"
INSTANCIAS_DIR = SCRIPT_DIR / "instancias_jssp"
EXPERIMENTO_TAG = "tres_casos_A_D"
VENDOR_DIR = SCRIPT_DIR / "vendor_py"

if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np

from generador_instancias_jssp import InstanciaJSSP, crear_casos_experimentales, guardar_casos_csv


@dataclass
class ResultadoCorrida:
    caso: str
    n_trabajos: int
    n_maquinas: int
    semilla_instancia: str
    configuracion: str
    semilla: int
    mejor_makespan: float
    tiempo_segundos: float
    soluciones_infactivas: int
    porcentaje_infactivas: float
    metodo_kmeans: str
    observacion: str
    historial: list[float]


def fijar_semilla(semilla: int) -> np.random.Generator:
    random.seed(semilla)
    np.random.seed(semilla)
    return np.random.default_rng(semilla)


def cruza_pmx(padreA: np.ndarray, padreB: np.ndarray, celdas: int) -> tuple[np.ndarray, np.ndarray]:
    """Cruza PMX reutilizada del script Python base T8/T9."""
    cortes = sorted(random.sample(range(1, celdas - 1), 2))
    inicio, fin = cortes[0], cortes[1]

    hijoA = np.full(celdas, -1)
    hijoB = np.full(celdas, -1)

    hijoA[inicio : fin + 1] = padreB[inicio : fin + 1]
    hijoB[inicio : fin + 1] = padreA[inicio : fin + 1]

    def rellenar_extremos(hijo: np.ndarray, p_origen: np.ndarray) -> np.ndarray:
        for c in range(celdas):
            if c < inicio or c > fin:
                val = p_origen[c]
                centro_hijo = hijo[inicio : fin + 1]
                while val in centro_hijo:
                    ind_relativo = np.where(centro_hijo == val)[0][0]
                    ind_real = inicio + ind_relativo
                    val = p_origen[ind_real]
                hijo[c] = val
        return hijo

    hijoA = rellenar_extremos(hijoA, padreA)
    hijoB = rellenar_extremos(hijoB, padreB)

    return hijoA, hijoB


def calcular_makespan(cromosoma: np.ndarray, tiempos: np.ndarray, secuencias: np.ndarray) -> float:
    """Funcion de makespan reutilizada del script Python base T8/T9."""
    tiempo_maq = np.zeros(cromosoma.shape[0])
    tiempo_trabajo = np.zeros(cromosoma.shape[1])
    paso_trabajo = np.zeros(cromosoma.shape[1], dtype=int)

    operaciones_pendientes = np.sum(tiempos > 0)

    while operaciones_pendientes > 0:
        asignacion_hecha = False
        for col in range(cromosoma.shape[1]):
            for m in range(cromosoma.shape[0]):
                trabajo = cromosoma[m, col] - 1
                if paso_trabajo[trabajo] < secuencias.shape[1]:
                    maq_requerida = secuencias[trabajo, paso_trabajo[trabajo]]
                    if maq_requerida != 0 and maq_requerida == (m + 1):
                        t_inicio = max(tiempo_maq[m], tiempo_trabajo[trabajo])
                        t_fin = t_inicio + tiempos[trabajo, m]

                        tiempo_maq[m] = t_fin
                        tiempo_trabajo[trabajo] = t_fin
                        paso_trabajo[trabajo] += 1
                        operaciones_pendientes -= 1
                        asignacion_hecha = True

        if not asignacion_hecha:
            break

    if operaciones_pendientes > 0:
        return float("inf")

    return float(np.max(tiempo_maq))


def crear_poblacion_aleatoria(
    n_individuos: int,
    n_maquinas: int,
    n_trabajos: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    poblacion = []
    trabajos = np.arange(1, n_trabajos + 1)
    for _ in range(n_individuos):
        cromosoma = np.array([rng.permutation(trabajos) for _ in range(n_maquinas)])
        poblacion.append(cromosoma)
    return poblacion


def calcular_perfil_carga_maquina(tiempos: np.ndarray) -> np.ndarray:
    """Perfil 1: cada trabajo queda descrito por sus tiempos por maquina."""
    return tiempos.astype(float)


def calcular_perfil_uso_maquinas(tiempos: np.ndarray) -> np.ndarray:
    """Perfil 2: cada trabajo queda descrito por las maquinas que usa."""
    return (tiempos > 0).astype(float)


def calcular_perfil(tiempos: np.ndarray, perfil_nombre: str) -> np.ndarray:
    if perfil_nombre == "perfil1_carga":
        return calcular_perfil_carga_maquina(tiempos)
    if perfil_nombre == "perfil2_uso":
        return calcular_perfil_uso_maquinas(tiempos)
    raise ValueError(f"Perfil no soportado: {perfil_nombre}")


def aplicar_kmeans(
    perfil: np.ndarray,
    k: int,
    semilla: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Aplica K-Means con la libreria scikit-learn."""
    try:
        from sklearn.cluster import KMeans  # type: ignore

        modelo = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=semilla)
        etiquetas = modelo.fit_predict(perfil)
        return etiquetas.astype(int), modelo.cluster_centers_, "sklearn.KMeans"
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "No se encontro scikit-learn. Instala scikit-learn o conserva la carpeta vendor_py "
            "generada para esta prueba."
        ) from exc


def ordenar_clusters_y_trabajos(
    tiempos: np.ndarray,
    secuencias: np.ndarray,
    etiquetas: np.ndarray,
    regla: str,
) -> list[int]:
    """Ordena clusters y trabajos segun la regla de despacho seleccionada."""
    total_por_trabajo = tiempos.sum(axis=1)
    operaciones_por_trabajo = (secuencias != 0).sum(axis=1)

    def clave_cluster(cluster: int) -> tuple[float, ...]:
        miembros = etiquetas == cluster
        carga_cluster = float(total_por_trabajo[miembros].sum())
        ops_cluster = float(operaciones_por_trabajo[miembros].sum())
        if regla == "SPT":
            return (carga_cluster, float(cluster))
        if regla in {"LPT", "MWR"}:
            return (-carga_cluster, float(cluster))
        if regla == "MOPNR":
            return (-ops_cluster, -carga_cluster, float(cluster))
        raise ValueError(f"Regla no soportada: {regla}")

    def clave_trabajo(idx: int) -> tuple[float, ...]:
        carga = float(total_por_trabajo[idx])
        ops = float(operaciones_por_trabajo[idx])
        if regla == "SPT":
            return (carga, float(idx))
        if regla in {"LPT", "MWR"}:
            return (-carga, float(idx))
        if regla == "MOPNR":
            return (-ops, -carga, float(idx))
        raise ValueError(f"Regla no soportada: {regla}")

    clusters = sorted(np.unique(etiquetas), key=lambda c: clave_cluster(int(c)))

    orden_base: list[int] = []
    for cluster in clusters:
        indices = np.where(etiquetas == cluster)[0]
        ordenados = sorted(indices, key=lambda idx: clave_trabajo(int(idx)))
        orden_base.extend(int(idx + 1) for idx in ordenados)

    return orden_base


def variante_controlada_orden(
    orden_base: list[int],
    rng: np.random.Generator,
    intensidad: int,
) -> list[int]:
    """Genera variantes sin romper la permutacion de trabajos."""
    orden = list(orden_base)
    if len(orden) < 2:
        return orden

    for _ in range(max(1, intensidad)):
        tipo = rng.choice(["rcl", "intercambio", "inversion", "rotacion"])

        if tipo == "rcl":
            orden = lista_restringida_candidatos(orden, rng, tamano=2)
        elif tipo == "intercambio":
            a, b = sorted(rng.choice(len(orden), size=2, replace=False))
            orden[a], orden[b] = orden[b], orden[a]
        elif tipo == "inversion":
            a, b = sorted(rng.choice(len(orden), size=2, replace=False))
            orden[a : b + 1] = reversed(orden[a : b + 1])
        else:
            paso = int(rng.integers(1, len(orden)))
            orden = orden[paso:] + orden[:paso]

    return orden


def lista_restringida_candidatos(
    orden: list[int],
    rng: np.random.Generator,
    tamano: int = 2,
) -> list[int]:
    pendientes = list(orden)
    resultado: list[int] = []
    while pendientes:
        limite = min(tamano, len(pendientes))
        elegido = int(rng.integers(0, limite))
        resultado.append(pendientes.pop(elegido))
    return resultado


def crear_cromosoma_desde_orden_base(
    orden_base: list[int],
    n_maquinas: int,
    rng: np.random.Generator,
    indice_individuo: int,
) -> np.ndarray:
    filas = []
    for maq in range(n_maquinas):
        intensidad = 1 + ((indice_individuo + maq) % 3)
        filas.append(variante_controlada_orden(orden_base, rng, intensidad))
    return np.array(filas, dtype=int)


def crear_individuos_kmeans(
    tiempos: np.ndarray,
    secuencias: np.ndarray,
    n_individuos: int,
    n_maquinas: int,
    k_clusters: int,
    semilla: int,
    rng: np.random.Generator,
    perfil_nombre: str,
    regla: str,
    indice_offset: int = 0,
) -> tuple[list[np.ndarray], str, list[int]]:
    perfil = calcular_perfil(tiempos, perfil_nombre)
    etiquetas, _, metodo = aplicar_kmeans(perfil, k_clusters, semilla)
    orden_base = ordenar_clusters_y_trabajos(tiempos, secuencias, etiquetas, regla)

    individuos = [
        crear_cromosoma_desde_orden_base(orden_base, n_maquinas, rng, indice_offset + indice)
        for indice in range(n_individuos)
    ]

    return individuos, metodo, orden_base


def crear_poblacion_configurada(
    configuracion: str,
    tiempos: np.ndarray,
    secuencias: np.ndarray,
    n_individuos: int,
    n_maquinas: int,
    n_trabajos: int,
    k_clusters: int,
    semilla: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], str, str]:
    componentes = {
        "B": [
            {"tipo": "km", "cantidad": 14, "perfil": "perfil1_carga", "regla": "SPT"},
            {"tipo": "aleatoria", "cantidad": 6},
        ],
        "C": [
            {"tipo": "km", "cantidad": 8, "perfil": "perfil1_carga", "regla": "SPT"},
            {"tipo": "km", "cantidad": 8, "perfil": "perfil2_uso", "regla": "SPT"},
            {"tipo": "aleatoria", "cantidad": 4},
        ],
        "D": [
            {"tipo": "km", "cantidad": 4, "perfil": "perfil1_carga", "regla": "SPT"},
            {"tipo": "km", "cantidad": 4, "perfil": "perfil1_carga", "regla": "LPT"},
            {"tipo": "km", "cantidad": 4, "perfil": "perfil2_uso", "regla": "MWR"},
            {"tipo": "km", "cantidad": 4, "perfil": "perfil2_uso", "regla": "MOPNR"},
            {"tipo": "aleatoria", "cantidad": 4},
        ],
    }

    if configuracion not in componentes:
        raise ValueError(f"Configuracion KM no soportada: {configuracion}")

    poblacion: list[np.ndarray] = []
    metodos: list[str] = []
    descripciones: list[str] = []

    for componente in componentes[configuracion]:
        cantidad = int(componente["cantidad"])
        if componente["tipo"] == "aleatoria":
            poblacion.extend(crear_poblacion_aleatoria(cantidad, n_maquinas, n_trabajos, rng))
            descripciones.append(f"{cantidad} aleatorios")
            continue

        perfil = str(componente["perfil"])
        regla = str(componente["regla"])
        individuos, metodo, orden_base = crear_individuos_kmeans(
            tiempos,
            secuencias,
            cantidad,
            n_maquinas,
            k_clusters,
            semilla,
            rng,
            perfil,
            regla,
            indice_offset=len(poblacion),
        )
        poblacion.extend(individuos)
        metodos.append(metodo)
        descripciones.append(f"{cantidad} KM {perfil} + {regla}; orden {orden_base}")

    if len(poblacion) != n_individuos:
        raise ValueError(f"La poblacion de {configuracion} tiene {len(poblacion)} individuos, no {n_individuos}.")

    metodo_kmeans = "+".join(sorted(set(metodos))) if metodos else "no_aplica"
    observacion = "; ".join(descripciones)
    return poblacion, metodo_kmeans, observacion


def cromosoma_es_valido(cromosoma: np.ndarray, n_trabajos: int) -> bool:
    esperado = list(range(1, n_trabajos + 1))
    return all(sorted(fila.tolist()) == esperado for fila in cromosoma)


def validar_poblacion(poblacion: list[np.ndarray], n_trabajos: int) -> None:
    invalidos = [idx for idx, cromosoma in enumerate(poblacion) if not cromosoma_es_valido(cromosoma, n_trabajos)]
    if invalidos:
        raise ValueError(f"Cromosomas invalidos en indices: {invalidos}")


def ejecutar_ag(
    poblacion_inicial: list[np.ndarray],
    tiempos: np.ndarray,
    secuencias: np.ndarray,
    semilla: int,
) -> tuple[float, np.ndarray | None, list[float], int, int]:
    fijar_semilla(semilla)
    poblacion = [np.copy(individuo) for individuo in poblacion_inicial]
    n_individuos = len(poblacion)
    n_maquinas = poblacion[0].shape[0]
    n_trabajos = poblacion[0].shape[1]

    mejor_historico_cromosoma = None
    mejor_historico_makespan = float("inf")
    historial: list[float] = []
    infactibles_totales = 0
    evaluaciones_totales = 0

    for _gen in range(1, M_GENERACIONES + 1):
        makespans = np.array([calcular_makespan(ind, tiempos, secuencias) for ind in poblacion])
        infactibles_totales += int(np.isinf(makespans).sum())
        evaluaciones_totales += len(makespans)

        idx_peor = int(np.argmax(makespans))
        idx_mejor = int(np.argmin(makespans))

        if makespans[idx_mejor] < mejor_historico_makespan:
            mejor_historico_makespan = float(makespans[idx_mejor])
            mejor_historico_cromosoma = np.copy(poblacion[idx_mejor])

        if _gen > 1 and mejor_historico_cromosoma is not None:
            poblacion[idx_peor] = np.copy(mejor_historico_cromosoma)
            makespans[idx_peor] = mejor_historico_makespan

        makespans_validos = makespans[np.isfinite(makespans)]
        max_valido = float(np.max(makespans_validos)) if len(makespans_validos) > 0 else 0.0

        fitness = np.zeros(n_individuos)
        for i in range(n_individuos):
            fitness[i] = 0.0 if not np.isfinite(makespans[i]) else max_valido - makespans[i]

        ganadores = []
        for _ in range(n_individuos):
            participantes = random.sample(range(n_individuos), K_TORNEO)
            idx_ganador = max(participantes, key=lambda idx: fitness[idx])
            ganadores.append(np.copy(poblacion[idx_ganador]))

        nueva_generacion = []
        limite_cruzas = math.floor((n_individuos * 0.95) / 2)

        for _ in range(limite_cruzas):
            idx_padres = random.sample(range(n_individuos), 2)
            padreA = ganadores[idx_padres[0]]
            padreB = ganadores[idx_padres[1]]

            hijoA_matriz = np.zeros((n_maquinas, n_trabajos), dtype=int)
            hijoB_matriz = np.zeros((n_maquinas, n_trabajos), dtype=int)

            for f in range(n_maquinas):
                hA, hB = cruza_pmx(padreA[f, :], padreB[f, :], n_trabajos)
                hijoA_matriz[f, :] = hA
                hijoB_matriz[f, :] = hB

            nueva_generacion.append(hijoA_matriz)
            nueva_generacion.append(hijoB_matriz)

        mientras_falten = n_individuos - len(nueva_generacion)
        if mientras_falten > 0:
            idx_clones = random.sample(range(n_individuos), mientras_falten)
            for clon in idx_clones:
                nueva_generacion.append(np.copy(ganadores[clon]))

        porcentaje_gen = _gen / M_GENERACIONES
        if porcentaje_gen <= 0.25:
            tasa_muta = 0.02
        elif porcentaje_gen <= 0.50:
            tasa_muta = 0.03
        elif porcentaje_gen <= 0.60:
            tasa_muta = 0.04
        else:
            tasa_muta = 0.05

        num_mutar = max(1, round(n_individuos * tasa_muta))
        idx_a_mutar = random.sample(range(n_individuos), num_mutar)

        for idx in idx_a_mutar:
            cromosoma_mutar = nueva_generacion[idx]
            for maq in range(n_maquinas):
                tipo = random.choice(["intercambio", "inversion"])
                pts = sorted(random.sample(range(n_trabajos), 2))

                if tipo == "intercambio":
                    temp = cromosoma_mutar[maq, pts[0]]
                    cromosoma_mutar[maq, pts[0]] = cromosoma_mutar[maq, pts[1]]
                    cromosoma_mutar[maq, pts[1]] = temp
                else:
                    cromosoma_mutar[maq, pts[0] : pts[1] + 1] = np.flip(
                        cromosoma_mutar[maq, pts[0] : pts[1] + 1]
                    )

            nueva_generacion[idx] = cromosoma_mutar

        validar_poblacion(nueva_generacion, n_trabajos)
        poblacion = nueva_generacion
        historial.append(mejor_historico_makespan)

    return mejor_historico_makespan, mejor_historico_cromosoma, historial, infactibles_totales, evaluaciones_totales


def ejecutar_corrida(
    instancia: InstanciaJSSP,
    configuracion: str,
    semilla: int,
) -> ResultadoCorrida:
    rng = fijar_semilla(semilla)

    if configuracion == "A":
        poblacion = crear_poblacion_aleatoria(
            N_INDIVIDUOS,
            instancia.n_maquinas,
            instancia.n_trabajos,
            rng,
        )
        metodo_kmeans = "no_aplica"
        observacion = "Poblacion 100% aleatoria"
    elif configuracion in {"B", "C", "D"}:
        poblacion, metodo_kmeans, observacion = crear_poblacion_configurada(
            configuracion,
            instancia.tiempos_ops,
            instancia.secuencia_ops,
            N_INDIVIDUOS,
            instancia.n_maquinas,
            instancia.n_trabajos,
            K_CLUSTERS,
            semilla,
            rng,
        )
    else:
        raise ValueError(f"Configuracion no soportada: {configuracion}")

    validar_poblacion(poblacion, instancia.n_trabajos)

    inicio = time.time()
    mejor_makespan, _, historial, infactibles_totales, evaluaciones_totales = ejecutar_ag(
        poblacion,
        instancia.tiempos_ops,
        instancia.secuencia_ops,
        semilla,
    )
    tiempo_segundos = time.time() - inicio

    porcentaje_infactivas = 100.0 * infactibles_totales / evaluaciones_totales
    if infactibles_totales > 0:
        observacion = f"{observacion}; infactibles evaluadas: {infactibles_totales}"

    return ResultadoCorrida(
        caso=instancia.nombre,
        n_trabajos=instancia.n_trabajos,
        n_maquinas=instancia.n_maquinas,
        semilla_instancia=instancia.semilla_instancia,
        configuracion=configuracion,
        semilla=semilla,
        mejor_makespan=mejor_makespan,
        tiempo_segundos=tiempo_segundos,
        soluciones_infactivas=infactibles_totales,
        porcentaje_infactivas=porcentaje_infactivas,
        metodo_kmeans=metodo_kmeans,
        observacion=observacion,
        historial=historial,
    )


def ejecutar_experimento(instancias: list[InstanciaJSSP]) -> list[ResultadoCorrida]:
    resultados = []
    for instancia in instancias:
        for configuracion in CONFIGURACIONES:
            for semilla in SEMILLAS:
                print(f"Ejecutando caso {instancia.nombre}, config {configuracion}, semilla {semilla}...")
                resultado = ejecutar_corrida(
                    instancia,
                    configuracion,
                    semilla,
                )
                resultados.append(resultado)
                print(
                    f"  Mejor makespan: {resultado.mejor_makespan:.0f} | "
                    f"tiempo: {resultado.tiempo_segundos:.3f}s | "
                    f"KMeans: {resultado.metodo_kmeans}"
                )
    return resultados


def guardar_resultados(resultados: list[ResultadoCorrida], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    resumen_path = output_dir / f"resultados_{EXPERIMENTO_TAG}.csv"
    with resumen_path.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(
            archivo,
            fieldnames=[
                "caso",
                "trabajos",
                "maquinas",
                "semilla_instancia",
                "configuracion",
                "semilla",
                "mejor_makespan",
                "tiempo_segundos",
                "soluciones_infactivas",
                "porcentaje_infactivas",
                "metodo_kmeans",
                "observacion",
            ],
        )
        writer.writeheader()
        for resultado in resultados:
            writer.writerow(
                {
                    "caso": resultado.caso,
                    "trabajos": resultado.n_trabajos,
                    "maquinas": resultado.n_maquinas,
                    "semilla_instancia": resultado.semilla_instancia,
                    "configuracion": resultado.configuracion,
                    "semilla": resultado.semilla,
                    "mejor_makespan": resultado.mejor_makespan,
                    "tiempo_segundos": round(resultado.tiempo_segundos, 6),
                    "soluciones_infactivas": resultado.soluciones_infactivas,
                    "porcentaje_infactivas": round(resultado.porcentaje_infactivas, 6),
                    "metodo_kmeans": resultado.metodo_kmeans,
                    "observacion": resultado.observacion,
                }
            )

    convergencia_path = output_dir / f"convergencia_{EXPERIMENTO_TAG}.csv"
    with convergencia_path.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(
            archivo,
            fieldnames=[
                "caso",
                "semilla_instancia",
                "configuracion",
                "semilla",
                "generacion",
                "mejor_makespan",
            ],
        )
        writer.writeheader()
        for resultado in resultados:
            for generacion, mejor in enumerate(resultado.historial, start=1):
                writer.writerow(
                    {
                        "caso": resultado.caso,
                        "semilla_instancia": resultado.semilla_instancia,
                        "configuracion": resultado.configuracion,
                        "semilla": resultado.semilla,
                        "generacion": generacion,
                        "mejor_makespan": mejor,
                    }
                )

    estadistico_path = output_dir / f"resumen_estadistico_{EXPERIMENTO_TAG}.csv"
    with estadistico_path.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(
            archivo,
            fieldnames=[
                "caso",
                "trabajos",
                "maquinas",
                "semilla_instancia",
                "configuracion",
                "mejor_makespan",
                "promedio_makespan",
                "desv_est_makespan",
                "tiempo_promedio_segundos",
                "soluciones_infactivas",
                "porcentaje_infactivas",
                "corridas",
            ],
        )
        writer.writeheader()
        grupos = sorted({(r.caso, r.configuracion) for r in resultados})
        for caso, configuracion in grupos:
            grupo = [r for r in resultados if r.caso == caso and r.configuracion == configuracion]
            makespans = np.array([r.mejor_makespan for r in grupo], dtype=float)
            tiempos = np.array([r.tiempo_segundos for r in grupo], dtype=float)
            infactibles = sum(r.soluciones_infactivas for r in grupo)
            total_evaluaciones = len(grupo) * N_INDIVIDUOS * M_GENERACIONES
            writer.writerow(
                {
                    "caso": grupo[0].caso,
                    "trabajos": grupo[0].n_trabajos,
                    "maquinas": grupo[0].n_maquinas,
                    "semilla_instancia": grupo[0].semilla_instancia,
                    "configuracion": configuracion,
                    "mejor_makespan": float(np.min(makespans)),
                    "promedio_makespan": float(np.mean(makespans)),
                    "desv_est_makespan": float(np.std(makespans, ddof=1)) if len(makespans) > 1 else 0.0,
                    "tiempo_promedio_segundos": round(float(np.mean(tiempos)), 6),
                    "soluciones_infactivas": infactibles,
                    "porcentaje_infactivas": round(100.0 * infactibles / total_evaluaciones, 6),
                    "corridas": len(grupo),
                }
            )


def imprimir_resumen(resultados: list[ResultadoCorrida]) -> None:
    print("\nResumen inicial tres casos A-D")
    print("caso,configuracion,semilla,mejor_makespan,tiempo_segundos,metodo_kmeans")
    for resultado in resultados:
        print(
            f"{resultado.caso},{resultado.configuracion},{resultado.semilla},"
            f"{resultado.mejor_makespan:.0f},{resultado.tiempo_segundos:.3f},"
            f"{resultado.metodo_kmeans}"
        )


def main() -> None:
    instancias = crear_casos_experimentales()
    guardar_casos_csv(instancias, INSTANCIAS_DIR)
    resultados = ejecutar_experimento(instancias)
    guardar_resultados(resultados, OUTPUT_DIR)
    imprimir_resumen(resultados)
    print(f"\nInstancias guardadas en: {INSTANCIAS_DIR}")
    print(f"\nArchivos guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
