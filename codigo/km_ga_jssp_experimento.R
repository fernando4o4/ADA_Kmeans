# ==============================================================================
# KM-GA PARA JSSP EN R - EXPERIMENTO FINAL A-D EN TRES CASOS
# ==============================================================================
# Replicación exacta del algoritmo KM-GA en R sin dependencias pesadas.
#
# Características de la réplica:
# - Carga de instancias experimentales desde archivos CSV (Base, Mediano, Grande).
# - Cálculo de Makespan adaptado a 1-based indexing de R.
# - Operador de Cruce PMX (Partially Mapped Crossover).
# - Clustering guiado por K-Means (usando la función base `kmeans`).
# - Reglas de despacho SPT, LPT, MWR y MOPNR para ordenar clusters y trabajos.
# - Operadores de variación: RCL, intercambio, inversión y rotación.
# - Algoritmo Genético principal con torneo, PMX, mutación dinámica y elitismo.
# ==============================================================================

# Constantes del experimento
N_INDIVIDUOS <- 20
M_GENERACIONES <- 50
K_TORNEO <- 3
K_CLUSTERS <- 3
SEMILLAS <- c(1, 2, 3)
CONFIGURACIONES <- c("A", "B", "C", "D")
EXPERIMENTO_TAG <- "tres_casos_A_D"

# ==============================================================================
# 1. CARGA Y VALIDACIÓN DE INSTANCIAS JSSP
# ==============================================================================

cargar_instancia <- function(nombre, n_trabajos, n_maquinas, max_operaciones, semilla_instancia, instancias_dir) {
  slug <- tolower(gsub(" ", "_", nombre))
  
  secuencia_path <- file.path(instancias_dir, paste0(slug, "_secuencia_ops.csv"))
  tiempos_path <- file.path(instancias_dir, paste0(slug, "_tiempos_ops.csv"))
  
  if (!file.exists(secuencia_path) || !file.exists(tiempos_path)) {
    stop(paste("No se encontraron los archivos CSV para la instancia:", nombre, 
               "\nBusque en:", secuencia_path, "y", tiempos_path))
  }
  
  # Leer CSVs. Omitir la primera columna de texto ("trabajo")
  secuencia_df <- read.csv(secuencia_path, header = TRUE, row.names = 1, check.names = FALSE)
  tiempos_df <- read.csv(tiempos_path, header = TRUE, row.names = 1, check.names = FALSE)
  
  secuencia_ops <- as.matrix(secuencia_df)
  tiempos_ops <- as.matrix(tiempos_df)
  
  # Limpiar nombres para evitar desajustes
  rownames(secuencia_ops) <- NULL
  colnames(secuencia_ops) <- NULL
  rownames(tiempos_ops) <- NULL
  colnames(tiempos_ops) <- NULL
  
  # Validar dimensiones
  if (nrow(secuencia_ops) != n_trabajos || ncol(secuencia_ops) != max_operaciones) {
    stop(paste("Dimensiones de secuencia invalidas para", nombre))
  }
  if (nrow(tiempos_ops) != n_trabajos || ncol(tiempos_ops) != n_maquinas) {
    stop(paste("Dimensiones de tiempos invalidas para", nombre))
  }
  
  # Validar integridad lógica del JSSP
  for (trabajo in 1:n_trabajos) {
    ruta <- secuencia_ops[trabajo, ]
    maquinas_usadas <- ruta[ruta > 0]
    n_operaciones <- length(maquinas_usadas)
    
    if (n_operaciones < 1 || n_operaciones > max_operaciones) {
      stop(paste("Cantidad de operaciones invalida en J", trabajo))
    }
    if (length(unique(maquinas_usadas)) != n_operaciones) {
      stop(paste("Una maquina se repite en la ruta de J", trabajo))
    }
    if (any(maquinas_usadas > n_maquinas)) {
      stop(paste("Ruta fuera del rango de maquinas en J", trabajo))
    }
    
    positivas <- which(tiempos_ops[trabajo, ] > 0)
    if (!all(sort(positivas) == sort(maquinas_usadas))) {
      stop(paste("La ruta y los tiempos de procesamiento no coinciden para J", trabajo))
    }
  }
  
  return(list(
    nombre = nombre,
    n_trabajos = n_trabajos,
    n_maquinas = n_maquinas,
    max_operaciones = max_operaciones,
    semilla_instancia = semilla_instancia,
    secuencia_ops = secuencia_ops,
    tiempos_ops = tiempos_ops
  ))
}

# ==============================================================================
# 2. OPERACIONES GENÉTICAS Y CÁLCULO DE MAKESPAN (1-based)
# ==============================================================================

calcular_makespan <- function(cromosoma, tiempos, secuencias) {
  n_maquinas <- nrow(cromosoma)
  n_trabajos <- ncol(cromosoma)
  
  tiempo_maq <- numeric(n_maquinas)
  tiempo_trabajo <- numeric(n_trabajos)
  paso_trabajo <- integer(n_trabajos) # Puntero al paso actual (0 = ninguna operacion completada)
  
  operaciones_pendientes <- sum(tiempos > 0)
  
  while (operaciones_pendientes > 0) {
    asignacion_hecha <- FALSE
    for (col in 1:n_trabajos) {
      for (m in 1:n_maquinas) {
        trabajo <- cromosoma[m, col]
        current_step <- paso_trabajo[trabajo] + 1
        
        if (current_step <= ncol(secuencias)) {
          maq_requerida <- secuencias[trabajo, current_step]
          if (maq_requerida != 0 && maq_requerida == m) {
            t_inicio <- max(tiempo_maq[m], tiempo_trabajo[trabajo])
            t_fin <- t_inicio + tiempos[trabajo, m]
            
            tiempo_maq[m] <- t_fin
            tiempo_trabajo[trabajo] <- t_fin
            paso_trabajo[trabajo] <- current_step
            operaciones_pendientes <- operaciones_pendientes - 1
            asignacion_hecha <- TRUE
          }
        }
      }
    }
    if (!asignacion_hecha) {
      break
    }
  }
  
  if (operaciones_pendientes > 0) {
    return(Inf)
  }
  
  return(max(tiempo_maq))
}

cruza_pmx <- function(padreA, padreB, celdas) {
  # Seleccionar dos puntos de corte distintos
  cortes <- sort(sample(2:(celdas - 1), 2))
  inicio <- cortes[1]
  fin <- cortes[2]
  
  hijoA <- rep(-1, celdas)
  hijoB <- rep(-1, celdas)
  
  hijoA[inicio:fin] <- padreB[inicio:fin]
  hijoB[inicio:fin] <- padreA[inicio:fin]
  
  rellenar_extremos <- function(hijo, p_origen) {
    centro_hijo <- hijo[inicio:fin]
    for (c in 1:celdas) {
      if (c < inicio || c > fin) {
        val <- p_origen[c]
        while (val %in% centro_hijo) {
          ind_relativo <- which(centro_hijo == val)[1]
          ind_real <- inicio + ind_relativo - 1
          val <- p_origen[ind_real]
        }
        hijo[c] <- val
      }
    }
    return(hijo)
  }
  
  hijoA <- rellenar_extremos(hijoA, padreA)
  hijoB <- rellenar_extremos(hijoB, padreB)
  
  return(list(hijoA, hijoB))
}

# ==============================================================================
# 3. K-MEANS Y DISEÑO DE PERFILES DE TRABAJOS
# ==============================================================================

calcular_perfil_carga_maquina <- function(tiempos) {
  return(matrix(as.numeric(tiempos), nrow = nrow(tiempos), ncol = ncol(tiempos)))
}

calcular_perfil_uso_maquinas <- function(tiempos) {
  return(matrix(as.numeric(tiempos > 0), nrow = nrow(tiempos), ncol = ncol(tiempos)))
}

calcular_perfil <- function(tiempos, perfil_nombre) {
  if (perfil_nombre == "perfil1_carga") {
    return(calcular_perfil_carga_maquina(tiempos))
  }
  if (perfil_nombre == "perfil2_uso") {
    return(calcular_perfil_uso_maquinas(tiempos))
  }
  stop(paste("Perfil no soportado:", perfil_nombre))
}

ordenar_clusters_y_trabajos <- function(tiempos, secuencias, etiquetas, regla) {
  total_por_trabajo <- rowSums(tiempos)
  operaciones_por_trabajo <- rowSums(secuencias != 0)
  
  unique_clusters <- unique(etiquetas)
  cluster_df <- data.frame(cluster = unique_clusters)
  cluster_df$carga <- sapply(cluster_df$cluster, function(c) sum(total_por_trabajo[etiquetas == c]))
  cluster_df$ops <- sapply(cluster_df$cluster, function(c) sum(operaciones_por_trabajo[etiquetas == c]))
  
  # Ordenar clusters lexicográficamente según la regla
  if (regla == "SPT") {
    cluster_order <- order(cluster_df$carga, cluster_df$cluster)
  } else if (regla %in% c("LPT", "MWR")) {
    cluster_order <- order(-cluster_df$carga, cluster_df$cluster)
  } else if (regla == "MOPNR") {
    cluster_order <- order(-cluster_df$ops, -cluster_df$carga, cluster_df$cluster)
  } else {
    stop(paste("Regla no soportada:", regla))
  }
  
  sorted_clusters <- cluster_df$cluster[cluster_order]
  
  # Ordenar trabajos dentro de cada cluster
  orden_base <- c()
  for (cl in sorted_clusters) {
    indices <- which(etiquetas == cl)
    if (length(indices) == 0) next
    
    job_df <- data.frame(idx = indices)
    job_df$carga <- total_por_trabajo[indices]
    job_df$ops <- operaciones_por_trabajo[indices]
    
    if (regla == "SPT") {
      job_order <- order(job_df$carga, job_df$idx)
    } else if (regla %in% c("LPT", "MWR")) {
      job_order <- order(-job_df$carga, job_df$idx)
    } else if (regla == "MOPNR") {
      job_order <- order(-job_df$ops, -job_df$carga, job_df$idx)
    } else {
      stop(paste("Regla no soportada:", regla))
    }
    
    orden_base <- c(orden_base, job_df$idx[job_order])
  }
  
  return(orden_base)
}

# ==============================================================================
# 4. OPERADORES DE VARIACIÓN CONTROLADA
# ==============================================================================

lista_restringida_candidatos <- function(orden, tamano = 2) {
  pendientes <- orden
  resultado <- c()
  while (length(pendientes) > 0) {
    limite <- min(tamano, length(pendientes))
    elegido <- sample(1:limite, 1)
    resultado <- c(resultado, pendientes[elegido])
    pendientes <- pendientes[-elegido]
  }
  return(resultado)
}

variante_controlada_orden <- function(orden_base, intensidad) {
  orden <- orden_base
  n <- length(orden)
  if (n < 2) return(orden)
  
  num_steps <- max(1, intensidad)
  for (step in 1:num_steps) {
    tipo <- sample(c("rcl", "intercambio", "inversion", "rotacion"), 1)
    
    if (tipo == "rcl") {
      orden <- lista_restringida_candidatos(orden, tamano = 2)
    } else if (tipo == "intercambio") {
      indices <- sort(sample(1:n, 2))
      a <- indices[1]
      b <- indices[2]
      temp <- orden[a]
      orden[a] <- orden[b]
      orden[b] <- temp
    } else if (tipo == "inversion") {
      indices <- sort(sample(1:n, 2))
      a <- indices[1]
      b <- indices[2]
      orden[a:b] <- rev(orden[a:b])
    } else if (tipo == "rotacion") {
      paso <- sample(1:(n - 1), 1)
      orden <- c(orden[(paso + 1):n], orden[1:paso])
    }
  }
  return(orden)
}

crear_cromosoma_desde_orden_base <- function(orden_base, n_maquinas, indice_individuo) {
  idx_0based <- indice_individuo - 1
  cromosoma <- matrix(0, nrow = n_maquinas, ncol = length(orden_base))
  for (m in 1:n_maquinas) {
    maq_0based <- m - 1
    # Intensidad dinamica de perturbación
    intensidad <- 1 + ((idx_0based + maq_0based) %% 3)
    cromosoma[m, ] <- variante_controlada_orden(orden_base, intensidad)
  }
  return(cromosoma)
}

# ==============================================================================
# 5. ESTRATEGIA DE INICIALIZACIÓN CONFIGURABLE
# ==============================================================================

crear_poblacion_aleatoria <- function(n_individuos, n_maquinas, n_trabajos) {
  poblacion <- list()
  for (i in 1:n_individuos) {
    cromosoma <- matrix(0, nrow = n_maquinas, ncol = n_trabajos)
    for (m in 1:n_maquinas) {
      cromosoma[m, ] <- sample(1:n_trabajos)
    }
    poblacion[[i]] <- cromosoma
  }
  return(poblacion)
}

crear_individuos_kmeans <- function(tiempos, secuencias, n_individuos, n_maquinas, k_clusters, semilla, perfil_nombre, regla, indice_offset = 0) {
  perfil <- calcular_perfil(tiempos, perfil_nombre)
  
  # Fijar semilla antes de ejecutar K-Means para garantizar reproducibilidad
  set.seed(semilla)
  fit <- kmeans(perfil, centers = k_clusters, nstart = 10)
  etiquetas <- fit$cluster
  
  orden_base <- ordenar_clusters_y_trabajos(tiempos, secuencias, etiquetas, regla)
  
  individuos <- list()
  for (i in 1:n_individuos) {
    individuos[[i]] <- crear_cromosoma_desde_orden_base(orden_base, n_maquinas, indice_offset + i)
  }
  
  return(list(individuos = individuos, metodo = "stats.kmeans", orden_base = orden_base))
}

crear_poblacion_configurada <- function(configuracion, tiempos, secuencias, n_individuos, n_maquinas, n_trabajos, k_clusters, semilla) {
  poblacion <- list()
  metodos <- c()
  descripciones <- c()
  
  if (configuracion == "B") {
    comps <- list(
      list(tipo = "km", cantidad = 14, perfil = "perfil1_carga", regla = "SPT"),
      list(tipo = "aleatoria", cantidad = 6)
    )
  } else if (configuracion == "C") {
    comps <- list(
      list(tipo = "km", cantidad = 8, perfil = "perfil1_carga", regla = "SPT"),
      list(tipo = "km", cantidad = 8, perfil = "perfil2_uso", regla = "SPT"),
      list(tipo = "aleatoria", cantidad = 4)
    )
  } else if (configuracion == "D") {
    comps <- list(
      list(tipo = "km", cantidad = 4, perfil = "perfil1_carga", regla = "SPT"),
      list(tipo = "km", cantidad = 4, perfil = "perfil1_carga", regla = "LPT"),
      list(tipo = "km", cantidad = 4, perfil = "perfil2_uso", regla = "MWR"),
      list(tipo = "km", cantidad = 4, perfil = "perfil2_uso", regla = "MOPNR"),
      list(tipo = "aleatoria", cantidad = 4)
    )
  } else {
    stop(paste("Configuracion KM no soportada:", configuracion))
  }
  
  for (comp in comps) {
    cantidad <- comp$cantidad
    if (comp$tipo == "aleatoria") {
      aleat <- crear_poblacion_aleatoria(cantidad, n_maquinas, n_trabajos)
      poblacion <- c(poblacion, aleat)
      descripciones <- c(descripciones, paste(cantidad, "aleatorios"))
    } else {
      perfil <- comp$perfil
      regla <- comp$regla
      res <- crear_individuos_kmeans(
        tiempos, secuencias, cantidad, n_maquinas, k_clusters, semilla,
        perfil, regla, indice_offset = length(poblacion)
      )
      poblacion <- c(poblacion, res$individuos)
      metodos <- c(metodos, res$metodo)
      orden_base_str <- paste(res$orden_base, collapse = ", ")
      descripciones <- c(descripciones, paste0(cantidad, " KM ", perfil, " + ", regla, "; orden [", orden_base_str, "]"))
    }
  }
  
  if (length(poblacion) != n_individuos) {
    stop(paste("La poblacion de", configuracion, "tiene", length(poblacion), "individuos, no", n_individuos))
  }
  
  unique_metodos <- sort(unique(metodos))
  metodo_kmeans <- if (length(unique_metodos) > 0) paste(unique_metodos, collapse = "+") else "no_aplica"
  observacion <- paste(descripciones, collapse = "; ")
  
  return(list(poblacion = poblacion, metodo_kmeans = metodo_kmeans, observacion = observacion))
}

cromosoma_es_valido <- function(cromosoma, n_trabajos) {
  esperado <- 1:n_trabajos
  for (r in 1:nrow(cromosoma)) {
    if (!all(sort(cromosoma[r, ]) == esperado)) {
      return(FALSE)
    }
  }
  return(TRUE)
}

validar_poblacion <- function(poblacion, n_trabajos) {
  invalidos <- c()
  for (i in 1:length(poblacion)) {
    if (!cromosoma_es_valido(poblacion[[i]], n_trabajos)) {
      invalidos <- c(invalidos, i)
    }
  }
  if (length(invalidos) > 0) {
    stop(paste("Cromosomas invalidos en indices:", paste(invalidos, collapse = ", ")))
  }
}

# ==============================================================================
# 6. ALGORITMO GENÉTICO PRINCIPAL
# ==============================================================================

ejecutar_ag <- function(poblacion_inicial, tiempos, secuencias, semilla) {
  set.seed(semilla)
  
  # Copiar poblacion inicial
  poblacion <- lapply(poblacion_inicial, function(x) matrix(x, nrow = nrow(x), ncol = ncol(x)))
  n_individuos <- length(poblacion)
  n_maquinas <- nrow(poblacion[[1]])
  n_trabajos <- ncol(poblacion[[1]])
  
  mejor_historico_cromosoma <- NULL
  mejor_historico_makespan <- Inf
  historial <- numeric(M_GENERACIONES)
  infactibles_totales <- 0
  evaluaciones_totales <- 0
  
  for (gen in 1:M_GENERACIONES) {
    makespans <- sapply(poblacion, function(ind) calcular_makespan(ind, tiempos, secuencias))
    infactibles_totales <- infactibles_totales + sum(is.infinite(makespans))
    evaluaciones_totales <- evaluaciones_totales + length(makespans)
    
    idx_peor <- which.max(makespans)
    idx_mejor <- which.min(makespans)
    
    if (makespans[idx_mejor] < mejor_historico_makespan) {
      mejor_historico_makespan <- makespans[idx_mejor]
      mejor_historico_cromosoma <- matrix(poblacion[[idx_mejor]], nrow = n_maquinas, ncol = n_trabajos)
    }
    
    # Elitismo: reemplazar el peor del turno actual con el mejor histórico
    if (gen > 1 && !is.null(mejor_historico_cromosoma)) {
      poblacion[[idx_peor]] <- matrix(mejor_historico_cromosoma, nrow = n_maquinas, ncol = n_trabajos)
      makespans[idx_peor] <- mejor_historico_makespan
    }
    
    makespans_validos <- makespans[is.finite(makespans)]
    max_valido <- if (length(makespans_validos) > 0) max(makespans_validos) else 0.0
    
    fitness <- numeric(n_individuos)
    for (i in 1:n_individuos) {
      fitness[i] <- if (is.infinite(makespans[i])) 0.0 else max_valido - makespans[i]
    }
    
    # Selección por torneo
    ganadores <- list()
    for (i in 1:n_individuos) {
      participantes <- sample(1:n_individuos, K_TORNEO)
      idx_ganador <- participantes[which.max(fitness[participantes])]
      ganadores[[i]] <- matrix(poblacion[[idx_ganador]], nrow = n_maquinas, ncol = n_trabajos)
    }
    
    # Reproducción
    nueva_generacion <- list()
    limite_cruzas <- floor((n_individuos * 0.95) / 2)
    
    for (i in 1:limite_cruzas) {
      idx_padres <- sample(1:n_individuos, 2)
      padreA <- ganadores[[idx_padres[1]]]
      padreB <- ganadores[[idx_padres[2]]]
      
      hijoA_matriz <- matrix(0, nrow = n_maquinas, ncol = n_trabajos)
      hijoB_matriz <- matrix(0, nrow = n_maquinas, ncol = n_trabajos)
      
      for (f in 1:n_maquinas) {
        res_cruza <- cruza_pmx(padreA[f, ], padreB[f, ], n_trabajos)
        hijoA_matriz[f, ] <- res_cruza[[1]]
        hijoB_matriz[f, ] <- res_cruza[[2]]
      }
      
      nueva_generacion[[length(nueva_generacion) + 1]] <- hijoA_matriz
      nueva_generacion[[length(nueva_generacion) + 1]] <- hijoB_matriz
    }
    
    # Clones para completar la población
    mientras_falten <- n_individuos - length(nueva_generacion)
    if (mientras_falten > 0) {
      idx_clones <- sample(1:n_individuos, mientras_falten)
      for (clon in idx_clones) {
        nueva_generacion[[length(nueva_generacion) + 1]] <- matrix(ganadores[[clon]], nrow = n_maquinas, ncol = n_trabajos)
      }
    }
    
    # Mutación dinámica
    porcentaje_gen <- gen / M_GENERACIONES
    if (porcentaje_gen <= 0.25) {
      tasa_muta <- 0.02
    } else if (porcentaje_gen <= 0.50) {
      tasa_muta <- 0.03
    } else if (porcentaje_gen <= 0.60) {
      tasa_muta <- 0.04
    } else {
      tasa_muta <- 0.05
    }
    
    num_mutar <- max(1, round(n_individuos * tasa_muta))
    idx_a_mutar <- sample(1:n_individuos, num_mutar)
    
    for (idx in idx_a_mutar) {
      cromosoma_mutar <- nueva_generacion[[idx]]
      for (maq in 1:n_maquinas) {
        tipo <- sample(c("intercambio", "inversion"), 1)
        pts <- sort(sample(1:n_trabajos, 2))
        
        if (tipo == "intercambio") {
          temp <- cromosoma_mutar[maq, pts[1]]
          cromosoma_mutar[maq, pts[1]] <- cromosoma_mutar[maq, pts[2]]
          cromosoma_mutar[maq, pts[2]] <- temp
        } else {
          cromosoma_mutar[maq, pts[1]:pts[2]] <- rev(cromosoma_mutar[maq, pts[1]:pts[2]])
        }
      }
      nueva_generacion[[idx]] <- cromosoma_mutar
    }
    
    validar_poblacion(nueva_generacion, n_trabajos)
    poblacion <- nueva_generacion
    historial[gen] <- mejor_historico_makespan
  }
  
  return(list(
    mejor_makespan = mejor_historico_makespan,
    mejor_cromosoma = mejor_historico_cromosoma,
    historial = historial,
    infactibles_totales = infactibles_totales,
    evaluaciones_totales = evaluaciones_totales
  ))
}

# ==============================================================================
# 7. EJECUCIÓN DEL EXPERIMENTO Y SALIDAS CSV
# ==============================================================================

ejecutar_corrida <- function(instancia, configuracion, semilla) {
  # Asegurar reproducibilidad de la inicializacion
  set.seed(semilla)
  
  if (configuracion == "A") {
    poblacion <- crear_poblacion_aleatoria(N_INDIVIDUOS, instancia$n_maquinas, instancia$n_trabajos)
    metodo_kmeans <- "no_aplica"
    observacion <- "Poblacion 100% aleatoria"
  } else if (configuracion %in% c("B", "C", "D")) {
    res_pop <- crear_poblacion_configurada(
      configuracion,
      instancia$tiempos_ops,
      instancia$secuencia_ops,
      N_INDIVIDUOS,
      instancia$n_maquinas,
      instancia$n_trabajos,
      K_CLUSTERS,
      semilla
    )
    poblacion <- res_pop$poblacion
    metodo_kmeans <- res_pop$metodo_kmeans
    observacion <- res_pop$observacion
  } else {
    stop(paste("Configuracion no soportada:", configuracion))
  }
  
  validar_poblacion(poblacion, instancia$n_trabajos)
  
  inicio <- Sys.time()
  res_ag <- ejecutar_ag(poblacion, instancia$tiempos_ops, instancia$secuencia_ops, semilla)
  tiempo_segundos <- as.numeric(difftime(Sys.time(), inicio, units = "secs"))
  
  porcentaje_infactivas <- 100.0 * res_ag$infactibles_totales / res_ag$evaluaciones_totales
  if (res_ag$infactibles_totales > 0) {
    observacion <- paste0(observacion, "; infactibles evaluadas: ", res_ag$infactibles_totales)
  }
  
  return(list(
    caso = instancia$nombre,
    n_trabajos = instancia$n_trabajos,
    n_maquinas = instancia$n_maquinas,
    semilla_instancia = instancia$semilla_instancia,
    configuracion = configuracion,
    semilla = semilla,
    mejor_makespan = res_ag$mejor_makespan,
    tiempo_segundos = tiempo_segundos,
    soluciones_infactivas = res_ag$infactibles_totales,
    porcentaje_infactivas = porcentaje_infactivas,
    metodo_kmeans = metodo_kmeans,
    observacion = observacion,
    historial = res_ag$historial
  ))
}

ejecutar_experimento <- function(instancias) {
  resultados <- list()
  for (instancia in instancias) {
    for (configuracion in CONFIGURACIONES) {
      for (semilla in SEMILLAS) {
        cat(paste0("Ejecutando caso ", instancia$nombre, ", config ", configuracion, ", semilla ", semilla, "...\n"))
        resultado <- ejecutar_corrida(instancia, configuracion, semilla)
        resultados[[length(resultados) + 1]] <- resultado
        cat(paste0("  Mejor makespan: ", resultado$mejor_makespan, 
                   " | tiempo: ", round(resultado$tiempo_segundos, 3), "s",
                   " | KMeans: ", resultado$metodo_kmeans, "\n"))
      }
    }
  }
  return(resultados)
}

guardar_resultados <- function(resultados, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # 1. CSV de Corridas Individuales
  resumen_path <- file.path(output_dir, paste0("resultados_", EXPERIMENTO_TAG, ".csv"))
  df_resumen <- data.frame(
    caso = sapply(resultados, function(r) r$caso),
    trabajos = sapply(resultados, function(r) r$n_trabajos),
    maquinas = sapply(resultados, function(r) r$n_maquinas),
    semilla_instancia = sapply(resultados, function(r) r$semilla_instancia),
    configuracion = sapply(resultados, function(r) r$configuracion),
    semilla = sapply(resultados, function(r) r$semilla),
    mejor_makespan = sapply(resultados, function(r) r$mejor_makespan),
    tiempo_segundos = round(sapply(resultados, function(r) r$tiempo_segundos), 6),
    soluciones_infactivas = sapply(resultados, function(r) r$soluciones_infactivas),
    porcentaje_infactivas = round(sapply(resultados, function(r) r$porcentaje_infactivas), 6),
    metodo_kmeans = sapply(resultados, function(r) r$metodo_kmeans),
    observacion = sapply(resultados, function(r) r$observacion),
    stringsAsFactors = FALSE
  )
  write.csv(df_resumen, resumen_path, row.names = FALSE, fileEncoding = "UTF-8")
  
  # 2. CSV de Convergencia
  convergencia_path <- file.path(output_dir, paste0("convergencia_", EXPERIMENTO_TAG, ".csv"))
  conv_list <- list()
  for (r in resultados) {
    for (gen in 1:length(r$historial)) {
      conv_list[[length(conv_list) + 1]] <- data.frame(
        caso = r$caso,
        semilla_instancia = r$semilla_instancia,
        configuracion = r$configuracion,
        semilla = r$semilla,
        generacion = gen,
        mejor_makespan = r$historial[gen],
        stringsAsFactors = FALSE
      )
    }
  }
  df_conv <- do.call(rbind, conv_list)
  write.csv(df_conv, convergencia_path, row.names = FALSE, fileEncoding = "UTF-8")
  
  # 3. CSV de Resumen Estadístico
  estadistico_path <- file.path(output_dir, paste0("resumen_estadistico_", EXPERIMENTO_TAG, ".csv"))
  grupos <- unique(lapply(resultados, function(r) list(caso = r$caso, configuracion = r$configuracion)))
  
  est_list <- list()
  for (g in grupos) {
    grupo_res <- Filter(function(r) r$caso == g$caso && r$configuracion == g$configuracion, resultados)
    makespans <- sapply(grupo_res, function(r) r$mejor_makespan)
    tiempos <- sapply(grupo_res, function(r) r$tiempo_segundos)
    infactibles <- sum(sapply(grupo_res, function(r) r$soluciones_infactivas))
    
    total_evaluaciones <- length(grupo_res) * N_INDIVIDUOS * M_GENERACIONES
    desv <- if (length(makespans) > 1) sd(makespans) else 0.0
    
    est_list[[length(est_list) + 1]] <- data.frame(
      caso = g$caso,
      trabajos = grupo_res[[1]]$n_trabajos,
      maquinas = grupo_res[[1]]$n_maquinas,
      semilla_instancia = grupo_res[[1]]$semilla_instancia,
      configuracion = g$configuracion,
      mejor_makespan = min(makespans),
      promedio_makespan = mean(makespans),
      desv_est_makespan = desv,
      tiempo_promedio_segundos = round(mean(tiempos), 6),
      soluciones_infactivas = infactibles,
      porcentaje_infactivas = round(100.0 * infactibles / total_evaluaciones, 6),
      corridas = length(grupo_res),
      stringsAsFactors = FALSE
    )
  }
  df_est <- do.call(rbind, est_list)
  write.csv(df_est, estadistico_path, row.names = FALSE, fileEncoding = "UTF-8")
}

imprimir_resumen <- function(resultados) {
  cat("\nResumen experimental de las corridas en R:\n")
  cat("caso,configuracion,semilla,mejor_makespan,tiempo_segundos,metodo_kmeans\n")
  for (r in resultados) {
    cat(paste0(r$caso, ",", r$configuracion, ",", r$semilla, ",", 
               round(r$mejor_makespan, 0), ",", round(r$tiempo_segundos, 3), ",", 
               r$metodo_kmeans, "\n"))
  }
}

# ==============================================================================
# 8. PUNTO DE ENTRADA PRINCIPAL (main)
# ==============================================================================

main <- function() {
  # Determinar dinámicamente directorios de entrada y salida
  current_dir <- getwd()
  instancias_dir <- ""
  
  if (dir.exists("instancias")) {
    instancias_dir <- file.path(current_dir, "instancias")
  } else if (dir.exists("../instancias")) {
    instancias_dir <- file.path(current_dir, "..", "instancias")
  } else {
    stop("No se encontro el directorio 'instancias' en './instancias' ni en '../instancias'. 
          Por favor, ejecute el script desde la carpeta raiz del proyecto o asegurese de que la carpeta 'instancias' este disponible.")
  }
  
  output_dir <- file.path(dirname(instancias_dir), "codigo", "resultados_r")
  
  cat(paste("Buscando instancias en:", instancias_dir, "\n"))
  cat(paste("Guardando resultados en:", output_dir, "\n\n"))
  
  # Cargar las 3 instancias ya existentes en el repositorio
  cat("Cargando instancias experimentales desde archivos CSV...\n")
  instancia_base <- cargar_instancia("Base", 8, 14, 5, "proporcionada", instancias_dir)
  instancia_mediana <- cargar_instancia("Mediano", 15, 14, 7, "1514", instancias_dir)
  instancia_grande <- cargar_instancia("Grande", 30, 20, 10, "3020", instancias_dir)
  
  instancias <- list(instancia_base, instancia_mediana, instancia_grande)
  
  # Ejecutar experimento consolidado
  resultados <- ejecutar_experimento(instancias)
  
  # Guardar los 3 CSVs correspondientes
  guardar_resultados(resultados, output_dir)
  
  # Mostrar el reporte resumido en consola
  imprimir_resumen(resultados)
  
  cat(paste0("\n¡Experimento KM-GA en R completado con exito!\n"))
  cat(paste0("Resultados CSV exportados en: ", output_dir, "\n"))
}

# Ejecutar el main si se llama directamente
if (!interactive()) {
  main()
}
