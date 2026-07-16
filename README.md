# Predicción de Abandono de Clientes (Customer Churn)

Sistema de predicción de abandono de clientes construido sobre datos de
telecomunicaciones, que combina análisis estadístico riguroso, modelado
supervisado e interpretabilidad con SHAP para generar recomendaciones
accionables de retención.

## Problema

Las empresas pierden ingresos cuando los clientes cancelan sus servicios.
Este proyecto responde una pregunta concreta: **¿qué clientes tienen mayor
riesgo de abandonar, y qué factores lo explican?**

## Metodología

1. Exploración y limpieza de datos
2. Análisis estadístico de asociación (Chi², Cramér's V, Mann-Whitney U)
3. Preprocesamiento con pipeline (One-Hot Encoding + StandardScaler)
4. Entrenamiento y comparación de modelos (Regresión Logística, Random Forest, XGBoost)
5. Interpretabilidad con SHAP (importancia global, dependencia, explicación individual)
6. Exportación del modelo final como `.pkl`

## Hallazgos principales

- **El tipo de contrato es el factor más determinante**: los clientes con
  contrato mes a mes tienen una probabilidad de abandono significativamente
  mayor que los de contrato anual o bianual.
- **La antigüedad protege contra el churn**: a mayor tiempo como cliente,
  menor riesgo de abandono. Los primeros 12 meses son el período más crítico.
- **Los cargos mensuales altos se asocian con mayor riesgo**, especialmente
  en combinación con contratos de corta duración.
- **La Regresión Logística fue seleccionada como modelo final** por su
  interpretabilidad y rendimiento competitivo frente a Random Forest y XGBoost.

## Implicación de negocio

El modelo permite priorizar campañas de retención hacia los segmentos de mayor
riesgo, optimizando el uso de recursos frente a estrategias de retención masiva.

## Herramientas

Python · pandas · scikit-learn · SHAP · matplotlib · seaborn · joblib

## Notebook

[Ver notebook completo →](./01_churn_prediction.ipynb)
