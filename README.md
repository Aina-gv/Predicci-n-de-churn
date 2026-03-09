

# **Predicción de Abandono de Clientes (Churn) – Red Neuronal Simple**

Este proyecto implementa una **red neuronal de una sola capa oculta** para predecir la **probabilidad de abandono de clientes** en una empresa de telecomunicaciones. Está construido completamente desde cero usando **NumPy**, sin librerías de machine learning externas.

Es un caso práctico orientado al entendimiento de:

*   Forward pass
*   Backpropagation
*   Función de activación sigmoide
*   Entrenamiento iterativo (gradient descent)
*   Evaluación y predicción con datos nuevos

***

# **Objetivo del proyecto**

Crear una red neuronal capaz de predecir si un cliente **abandonará** (churn = 1) o **se quedará** (churn = 0), usando solo:

*   Edad
*   Meses contratado
*   Uso mensual (GB)
*   Factura promedio (€)
*   Número de llamadas al soporte

La idea es llevar paso a paso el proceso de implementación, desde preparar los datos hasta obtener predicciones reales.

***

# **1. Preparación de los datos**

Los datos se simulan en un array con 5 características por cliente:

```python
X_raw = np.array([
    [34, 1, 5, 50, 0],
    [55, 48, 20, 70, 1],
    [23, 3, 50, 90, 8],
    [40, 6, 15, 65, 4],
    [60, 120, 10, 40, 1],
    [28, 2, 45, 85, 6]
])
```

Las etiquetas son 0 = no abandona / 1 = abandona.

Para asegurar buen rendimiento, se aplicó **normalización Min-Max**:

```python
X = X_raw / X_raw.max(axis=0)
```

Esto evita que variables como *meses contratado* (rango 1–120) dominen sobre otras como *soporte llamado* (0–8).

***

# **2. Implementación de la red neuronal**

### Arquitectura escogida

*   **5 neuronas de entrada**
*   **1 capa oculta de 4 neuronas** (activación sigmoide)
*   **1 neurona de salida** (sigmoide → probabilidad de abandono)

### Función sigmoide y derivada

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)
```


### Inicialización de pesos

```python
w1 = np.random.randn(5, 4)
b1 = np.random.randn(1, 4)
w2 = np.random.randn(4, 1)
b2 = np.random.randn(1, 1)
```

### Forward pass + Backpropagation

El entrenamiento sigue el flujo clásico:

1.  Capa oculta
2.  Capa salida
3.  Cálculo del error
4.  Backpropagation
5.  Actualización de pesos

Todo implementado a mano, sin frameworks externos.

Ejemplo del forward:

```python
z1 = np.dot(X, w1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, w2) + b2
a2 = sigmoid(z2)
```

***

# **3. Entrenamiento del modelo**

*   **Tasa de aprendizaje:** 0.1
*   **Épocas:** 10.000
*   **Función de pérdida:** Error Cuadrático Medio

El modelo imprime el progreso cada 2000 iteraciones.

```python
if epoch % 2000 == 0:
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

***

# **4. Evaluación del modelo**

Una vez entrenado, el modelo predice sobre los mismos datos de entrenamiento.

Predicciones finales:

```python
predicciones = np.round(a2)
precision = np.mean(predicciones == y) * 100
```

El modelo obtiene:

### **Precisión: 100% sobre el conjunto de entrenamiento**

***

# **5. Pruebas con nuevos clientes**

Se crean dos nuevos ejemplos:

```python
X_test_raw = np.array([
    [25, 2, 10, 55, 1],
    [30, 5, 60, 95, 7]
])
```

```python
a_test = sigmoid(np.dot(a1_test, w2) + b2)
```

### Resultado interpretado:

*   **Cliente A:** baja probabilidad de abandono
*   **Cliente B:** alta probabilidad de abandono

Impreso así:

    Cliente A: XX% prob. de abandono
    Cliente B: XX% prob. de abandono

***

# **Cómo trabajé el proyecto**

Para este trabajo:

*   Implementé todo paso a paso, desde cero
*   Validé que la normalización fuera correcta
*   Organicé el código en bloques muy claros (datos → red → entrenamiento → pruebas)
*   Usé sigmoide para salida por tratarse de clasificación binaria
*   Revisé los gradientes manualmente para asegurar correcto backpropagation

La estructura está pensada para ser **didáctica, clara y extensible**.

***

# **Dificultades y cómo las resolví**

Entender bien cómo propagar errores en dos capas  
Manejar dimensiones de matrices en NumPy  
Ajustar el learning rate para evitar saltarse mínimos  
Mantener la normalización igual en train y test  
Verificar que la función sigmoide no saturara demasiado

***

# **Conclusión**

Este proyecto permite comprender en profundidad:

*   Cómo funciona internamente una red neuronal
*   Cómo se calcula la función sigmoide y sus derivadas
*   Cómo se optimizan pesos con descenso de gradiente
*   Cómo se predice churn de forma sencilla

