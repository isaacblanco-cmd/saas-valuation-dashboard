
# SaaS Valuation Dashboard (Streamlit)

## 🚀 Cómo ejecutarlo en tu ordenador
1) Crea y activa un entorno:
```bash
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
```
2) Instala dependencias:
```bash
pip install -r requirements.txt
```
3) Lanza la app:
```bash
streamlit run app.py
```

## ☁️ Desplegar en Streamlit Cloud
1) Sube `app.py` y `requirements.txt` a un repositorio de GitHub.
2) Ve a https://share.streamlit.io (Streamlit Community Cloud).
3) Conecta tu repo y selecciona el archivo `app.py` como **Main file path**.
4) (Opcional) Añade tu Excel al repo o súbelo cada vez desde la app.

## 📥 Datos
- La app acepta CSV o Excel con columnas de **Fecha** y **ARR** o **MRR**.
- Si no aportas **Crecimiento_mensual** y **Churn_mensual**, se estima el crecimiento por `pct_change()` y churn = 0.

## 🧮 Cálculo de la valoración
- Se usa **múltiplo sobre ARR proyectado**: `Valoración = ARR_{proyectado} × múltiplo`.
- Proyección por meses: `ARR_{t+1} = ARR_t × (1 + crecimiento_mensual - churn_mensual)`.
- Escenarios: Base, Optimista (+20% crecimiento, -25% churn), Pesimista (-20% crecimiento, +25% churn).

