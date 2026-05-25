from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class InstanciaJSSP:
    nombre: str
    n_trabajos: int
    n_maquinas: int
    min_operaciones: int
    max_operaciones: int
    semilla_instancia: str
    secuencia_ops: np.ndarray
    tiempos_ops: np.ndarray


def validar_instancia(instancia: InstanciaJSSP) -> None:
    if instancia.secuencia_ops.shape != (instancia.n_trabajos, instancia.max_operaciones):
        raise ValueError(f"Forma de secuencia_ops invalida para {instancia.nombre}.")
    if instancia.tiempos_ops.shape != (instancia.n_trabajos, instancia.n_maquinas):
        raise ValueError(f"Forma de tiempos_ops invalida para {instancia.nombre}.")

    for trabajo in range(instancia.n_trabajos):
        ruta = instancia.secuencia_ops[trabajo]
        maquinas_usadas = ruta[ruta > 0]
        n_operaciones = len(maquinas_usadas)
        if not instancia.min_operaciones <= n_operaciones <= instancia.max_operaciones:
            raise ValueError(f"Cantidad de operaciones invalida en J{trabajo + 1}.")
        if len(set(maquinas_usadas.tolist())) != n_operaciones:
            raise ValueError(f"Una maquina se repite en la ruta de J{trabajo + 1}.")
        if np.any(maquinas_usadas > instancia.n_maquinas):
            raise ValueError(f"Ruta fuera del rango de maquinas en J{trabajo + 1}.")

        positivas = np.where(instancia.tiempos_ops[trabajo] > 0)[0] + 1
        if set(positivas.tolist()) != set(maquinas_usadas.tolist()):
            raise ValueError(f"La ruta y tiempos no coinciden para J{trabajo + 1}.")


def crear_instancia_base() -> InstanciaJSSP:
    secuencia_ops = np.array(
        [
            [1, 3, 0, 0, 0],
            [7, 13, 5, 1, 3],
            [7, 14, 5, 9, 0],
            [6, 0, 0, 0, 0],
            [10, 0, 0, 0, 0],
            [1, 3, 0, 0, 0],
            [6, 12, 11, 0, 0],
            [11, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    tiempos_ops = np.array(
        [
            [30, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [120, 0, 30, 0, 20, 0, 30, 0, 0, 0, 0, 0, 30, 0],
            [0, 0, 0, 0, 20, 0, 30, 0, 90, 0, 0, 0, 0, 30],
            [0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0],
            [120, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 20, 20, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0],
        ],
        dtype=int,
    )
    instancia = InstanciaJSSP(
        nombre="Base",
        n_trabajos=8,
        n_maquinas=14,
        min_operaciones=1,
        max_operaciones=5,
        semilla_instancia="proporcionada",
        secuencia_ops=secuencia_ops,
        tiempos_ops=tiempos_ops,
    )
    validar_instancia(instancia)
    return instancia


def generar_instancia(
    nombre: str,
    n_trabajos: int,
    n_maquinas: int,
    min_operaciones: int,
    max_operaciones: int,
    semilla: int,
) -> InstanciaJSSP:
    rng = np.random.default_rng(semilla)
    secuencia_ops = np.zeros((n_trabajos, max_operaciones), dtype=int)
    tiempos_ops = np.zeros((n_trabajos, n_maquinas), dtype=int)

    for trabajo in range(n_trabajos):
        n_operaciones = int(rng.integers(min_operaciones, max_operaciones + 1))
        ruta = rng.choice(np.arange(1, n_maquinas + 1), size=n_operaciones, replace=False)
        tiempos = rng.integers(10, 121, size=n_operaciones)

        secuencia_ops[trabajo, :n_operaciones] = ruta
        for maquina, tiempo in zip(ruta, tiempos):
            tiempos_ops[trabajo, maquina - 1] = tiempo

    instancia = InstanciaJSSP(
        nombre=nombre,
        n_trabajos=n_trabajos,
        n_maquinas=n_maquinas,
        min_operaciones=min_operaciones,
        max_operaciones=max_operaciones,
        semilla_instancia=str(semilla),
        secuencia_ops=secuencia_ops,
        tiempos_ops=tiempos_ops,
    )
    validar_instancia(instancia)
    return instancia


def crear_casos_experimentales() -> list[InstanciaJSSP]:
    return [
        crear_instancia_base(),
        generar_instancia("Mediano", 15, 14, 5, 7, semilla=1514),
        generar_instancia("Grande", 30, 20, 8, 10, semilla=3020),
    ]


def _slug(nombre: str) -> str:
    return nombre.lower().replace(" ", "_")


def guardar_instancia_csv(instancia: InstanciaJSSP, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(instancia.nombre)

    secuencia_path = output_dir / f"{slug}_secuencia_ops.csv"
    with secuencia_path.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["trabajo"] + [f"op_{i}" for i in range(1, instancia.max_operaciones + 1)])
        for idx, ruta in enumerate(instancia.secuencia_ops, start=1):
            writer.writerow([f"J{idx}"] + ruta.tolist())

    tiempos_path = output_dir / f"{slug}_tiempos_ops.csv"
    with tiempos_path.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["trabajo"] + [f"M{i}" for i in range(1, instancia.n_maquinas + 1)])
        for idx, tiempos in enumerate(instancia.tiempos_ops, start=1):
            writer.writerow([f"J{idx}"] + tiempos.tolist())

    metadata_path = output_dir / f"{slug}_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["parametro", "valor"])
        writer.writerow(["caso", instancia.nombre])
        writer.writerow(["trabajos", instancia.n_trabajos])
        writer.writerow(["maquinas", instancia.n_maquinas])
        writer.writerow(["operaciones_minimas", instancia.min_operaciones])
        writer.writerow(["operaciones_maximas", instancia.max_operaciones])
        writer.writerow(["tiempo_minimo", 10])
        writer.writerow(["tiempo_maximo", 120])
        writer.writerow(["semilla_instancia", instancia.semilla_instancia])


def guardar_casos_csv(instancias: list[InstanciaJSSP], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for instancia in instancias:
        validar_instancia(instancia)
        guardar_instancia_csv(instancia, output_dir)

    manifest_path = output_dir / "manifest_instancias.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(
            ["caso", "trabajos", "maquinas", "operaciones_minimas", "operaciones_maximas", "semilla_instancia"]
        )
        for instancia in instancias:
            writer.writerow(
                [
                    instancia.nombre,
                    instancia.n_trabajos,
                    instancia.n_maquinas,
                    instancia.min_operaciones,
                    instancia.max_operaciones,
                    instancia.semilla_instancia,
                ]
            )


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "instancias_jssp"
    instancias = crear_casos_experimentales()
    guardar_casos_csv(instancias, output_dir)
    for instancia in instancias:
        print(
            f"{instancia.nombre}: {instancia.n_trabajos} trabajos x "
            f"{instancia.n_maquinas} maquinas | semilla={instancia.semilla_instancia}"
        )
    print(f"Instancias guardadas en: {output_dir}")


if __name__ == "__main__":
    main()
