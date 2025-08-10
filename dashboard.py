import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import datetime
import re

warnings.filterwarnings('ignore')

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dashboard de Análisis de Ventas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS para mejorar la apariencia ---
st.markdown("""
<style>
    /* Estilo general del cuerpo */
    .main {
        background-color: #f0f2f6;
    }
    /* Estilo para las tarjetas de métricas */
    .stMetric {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Títulos de las métricas */
    .stMetric .st-ax {
        font-size: 1.2rem !important;
        font-weight: bold;
        color: #4a4a4a;
    }
    /* Valores de las métricas */
    .stMetric .st-c5 {
        font-size: 2.5rem !important;
        color: #0068c9;
    }
    /* Contenedores de los gráficos */
    .stPlotlyChart {
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: #FFFFFF;
    }
    /* Títulos de los gráficos */
    h2, h3 {
        color: #0068c9;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# --- Función para Cargar y Procesar los Datos (con caché para eficiencia) ---
@st.cache_data
def load_and_process_data(uploaded_file):
    """
    Carga los datos, los limpia (excluyendo CodPro 1), y genera un reporte detallado 
    junto con dataframes de las filas que contienen errores para su descarga.
    """
    try:
        # Intenta leer con la codificación estándar UTF-8 primero
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        # Si UTF-8 falla, intenta con 'latin-1' sin mostrar advertencia al usuario.
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin-1')
    except Exception as e:
        return None, f"Error fatal al leer el archivo: {e}. Asegúrate de que es un archivo CSV válido.", {}

    # --- Diagnóstico del Proceso de Limpieza ---
    initial_rows = len(df)
    report_log = [f"Archivo cargado con {initial_rows} filas iniciales."]
    error_dfs = {}

    # --- Excluir 'Impuesto A La Bolsa' (CodPro 1) ---
    # Asegurarse de que CodPro es numérico para una comparación segura
    df['CodPro'] = pd.to_numeric(df['CodPro'], errors='coerce')
    rows_before_exclusion = len(df)
    # Excluir filas donde CodPro es 1 y crear una copia para evitar advertencias
    df = df[df['CodPro'] != 1].copy()
    rows_after_exclusion = len(df)
    excluded_tax_rows = rows_before_exclusion - rows_after_exclusion
    if excluded_tax_rows > 0:
        report_log.append(f"Se excluyeron {excluded_tax_rows} filas correspondientes al 'Impuesto A La Bolsa' (CodPro 1).")
    
    # --- LIMPIEZA Y TRANSFORMACIÓN DE DATOS ---
    
    # Consolidar Grupos de Página Web
    web_groups = ['Pagina Web', 'Coordinadora', 'Contraentrega']
    df['Grupos'] = df['Grupos'].replace(web_groups, 'Pagina Web')
    report_log.append("Se consolidaron los grupos 'Coordinadora' y 'Contraentrega' en 'Pagina Web'.")

    # Columnas a convertir a tipo numérico
    numeric_cols = ['Venta Bruta', 'Descto', 'Valiva', 'Venta Neta', 'Costo', 'Utilidad Bruta', 'Cantid']
    
    for col in numeric_cols:
        # Identificar filas con valores no numéricos antes de la conversión
        is_error = pd.to_numeric(df[col], errors='coerce').isna()
        non_numeric_count = is_error.sum()
        
        if non_numeric_count > 0:
            report_log.append(f"En la columna '{col}', se encontraron {non_numeric_count} valores no numéricos que fueron convertidos a 0.")
            # Guardar las filas con error para su posterior descarga
            error_dfs[col] = df[is_error].copy()
        
        # Convertir la columna a numérico, rellenando errores y NaNs con 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Conversión de la columna 'Mesano' a formato de fecha
    df['Fecha'] = pd.to_datetime(df['Mesano'].astype(str).str.zfill(4), format='%y%m', errors='coerce')
    
    # Contar y reportar filas con fechas inválidas antes de eliminarlas
    invalid_dates_count = df['Fecha'].isna().sum()
    if invalid_dates_count > 0:
        report_log.append(f"Se encontraron y descartaron {invalid_dates_count} filas por tener un formato de fecha inválido en la columna 'Mesano'.")
        # Guardar filas con fechas inválidas para descarga
        error_dfs['Fecha_Invalida'] = df[df['Fecha'].isna()].copy()
        df.dropna(subset=['Fecha'], inplace=True)

    final_rows = len(df)
    rows_dropped = initial_rows - final_rows
    report_log.append(f"Proceso finalizado. Se procesaron {final_rows} filas. Total de filas descartadas (impuesto + fechas inválidas): {rows_dropped}.")

    # Crear columna 'Periodo' para mostrar en la tabla
    df['Periodo'] = df['Fecha'].dt.strftime('%Y-%m')

    # Renombrar columnas para mayor claridad ANTES de los cálculos
    df.rename(columns={
        'Nomsuc': 'Sucursal',
        'Detall': 'Producto',
        'Cantid': 'Cantidad',
        'Tipo Proveedor': 'Tipo_Proveedor',
        'Grupos': 'Grupo' # Renombrar para consistencia
    }, inplace=True)

    # --- Usar vectorización para un cálculo más eficiente ---
    # Se usa .div() para la división y se manejan los casos de división por cero
    # que resultan en infinito (inf) o NaN (Not a Number).
    df['Precio/Unidad'] = df['Venta Neta'].div(df['Cantidad']).replace([float('inf'), -float('inf')], 0).fillna(0)
    df['Costo/Unidad'] = df['Costo'].div(df['Cantidad']).replace([float('inf'), -float('inf')], 0).fillna(0)
    df['Margen Utilidad (%)'] = (df['Utilidad Bruta'].div(df['Venta Neta']) * 100).replace([float('inf'), -float('inf')], 0).fillna(0)

    return df, "\n".join(report_log), error_dfs

# --- TÍTULO PRINCIPAL DEL DASHBOARD ---
st.title("📊 Dashboard de Análisis de Ventas")
st.markdown("Sube tu archivo de ventas en formato CSV para explorar los datos de forma interactiva.")

# --- CARGADOR DE ARCHIVOS ---
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file is None:
    st.info("Por favor, carga un archivo CSV para comenzar el análisis.")
else:
    df, report, error_dfs = load_and_process_data(uploaded_file)

    # --- Mostrar el reporte de carga de datos ---
    with st.expander("Ver Resumen del Proceso de Carga de Datos"):
        st.info("Resumen del Proceso de Carga y Limpieza:")
        st.text(report)

        # --- Sección para descargar archivos de error ---
        if error_dfs:
            st.markdown("#### 📄 Archivos de Diagnóstico de Errores")
            st.warning("Se encontraron filas con datos en formato incorrecto. Descarga los archivos para revisarlos.")
            for col_name, error_df in error_dfs.items():
                st.download_button(
                    label=f"📥 Descargar filas con errores en '{col_name}' ({len(error_df)} filas)",
                    data=error_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'errores_{col_name.lower().replace(" ", "_")}.csv',
                    mime='text/csv',
                    key=f'download_{col_name}'
                )
        else:
            st.success("✅ ¡Excelente! No se encontraron errores de formato en los datos.")


    if df is not None and not df.empty:
        # --- BARRA LATERAL DE FILTROS ---
        st.sidebar.header("Filtros Interactivos")

        # --- El filtro de fecha ahora es el primero ---
        st.sidebar.markdown("##### Seleccionar Rango de Fechas")
        min_date = df['Fecha'].min().date()
        max_date = df['Fecha'].max().date()
        
        # Fecha inicial por defecto es 2024
        default_start_date = datetime.date(2024, 1, 1)
        default_end_date = datetime.date(2024, 12, 31)

        # Asegurarse de que el default no esté fuera de los límites de los datos
        if default_start_date < min_date:
            default_start_date = min_date
        if default_end_date > max_date:
            default_end_date = max_date

        selected_date_range = st.sidebar.date_input(
            "Rango de Fechas", value=(default_start_date, default_end_date),
            min_value=min_date, max_value=max_date, format="YYYY-MM-DD"
        )

        # Filtrar por fecha primero para que los otros filtros se actualicen
        if len(selected_date_range) == 2:
            start_date = pd.to_datetime(selected_date_range[0])
            end_date = pd.to_datetime(selected_date_range[1])
            df_by_date = df[(df['Fecha'] >= start_date) & (df['Fecha'] <= end_date)]
        else:
            df_by_date = df.copy() # Si no hay rango válido, usar todos los datos
        
        # --- Lógica para filtros en cascada ---
        
        # Filtro por Sucursal
        st.sidebar.markdown("##### Filtro por Sucursal")
        sucursales_disponibles = sorted(df_by_date['Sucursal'].unique())
        select_all_sucursales = st.sidebar.checkbox("Seleccionar Todas las Sucursales", value=True, key='all_sucursales')
        if select_all_sucursales:
            selected_sucursales = st.sidebar.multiselect("Sucursal", options=sucursales_disponibles, default=sucursales_disponibles)
        else:
            selected_sucursales = st.sidebar.multiselect("Sucursal", options=sucursales_disponibles)
        
        df_after_sucursal = df_by_date[df_by_date['Sucursal'].isin(selected_sucursales)]

        # Filtro por Grupo
        st.sidebar.markdown("##### Filtro por Grupo")
        grupos_disponibles = sorted(df_after_sucursal['Grupo'].unique())
        select_all_grupos = st.sidebar.checkbox("Seleccionar Todos los Grupos", value=True, key='all_grupos')
        if select_all_grupos:
            selected_grupos = st.sidebar.multiselect("Grupo", options=grupos_disponibles, default=grupos_disponibles)
        else:
            selected_grupos = st.sidebar.multiselect("Grupo", options=grupos_disponibles)

        df_after_grupo = df_after_sucursal[df_after_sucursal['Grupo'].isin(selected_grupos)]

        # Filtro por Tipo de Proveedor
        st.sidebar.markdown("##### Filtro por Tipo de Proveedor")
        proveedores_disponibles = sorted(df_after_grupo['Tipo_Proveedor'].unique())
        select_all_proveedores = st.sidebar.checkbox("Seleccionar Todos los Proveedores", value=True, key='all_proveedores')
        if select_all_proveedores:
            selected_proveedores = st.sidebar.multiselect("Tipo de Proveedor", options=proveedores_disponibles, default=proveedores_disponibles)
        else:
            selected_proveedores = st.sidebar.multiselect("Tipo de Proveedor", options=proveedores_disponibles)

        df_after_proveedor = df_after_grupo[df_after_grupo['Tipo_Proveedor'].isin(selected_proveedores)]

        # Filtro por Marca
        st.sidebar.markdown("##### Filtro por Marca")
        marcas_disponibles = sorted(df_after_proveedor['Marca'].unique())
        select_all_marcas = st.sidebar.checkbox("Seleccionar Todas las Marcas", value=True, key='all_marcas')
        if select_all_marcas:
            selected_marcas = st.sidebar.multiselect("Marca", options=marcas_disponibles, default=marcas_disponibles)
        else:
            selected_marcas = st.sidebar.multiselect("Marca", options=marcas_disponibles)
        
        # Buscador de SKU
        st.sidebar.markdown("##### Buscar por SKU (CodPro)")
        sku_input = st.sidebar.text_area("Ingresar SKUs (separados por coma, espacio o nueva línea)")


        # --- APLICACIÓN DE FILTROS AL DATAFRAME ---
        # El filtrado final se hace sobre el dataframe original para asegurar la integridad
        df_filtered = df[
            (df['Sucursal'].isin(selected_sucursales)) &
            (df['Grupo'].isin(selected_grupos)) &
            (df['Marca'].isin(selected_marcas)) &
            (df['Tipo_Proveedor'].isin(selected_proveedores)) &
            (df['Fecha'] >= start_date) &
            (df['Fecha'] <= end_date)
        ]
        
        # Aplicar filtro de SKU si el usuario ingresó algo
        if sku_input.strip():
            # Usar regex para separar por comas, espacios o saltos de línea
            skus_to_filter = [int(s.strip()) for s in re.split('[ ,\n]+', sku_input) if s.strip().isdigit()]
            if skus_to_filter:
                df_filtered = df_filtered[df_filtered['CodPro'].isin(skus_to_filter)]

        df_filtered = df_filtered.reset_index(drop=True)


        if df_filtered.empty:
            st.warning("No se encontraron datos para los filtros seleccionados. Por favor, ajusta tu selección.")
        else:
            # --- VISTA PRINCIPAL ---
            st.markdown("## 📈 Resumen General del Rendimiento")

            # --- KPIs con Comparativo Anual ---
            
            # Calcular periodo anterior y filtrar
            prev_start_date = start_date - pd.DateOffset(years=1)
            prev_end_date = end_date - pd.DateOffset(years=1)
            
            # Usar los mismos filtros categóricos para la comparación
            base_filters = (
                (df['Sucursal'].isin(selected_sucursales)) &
                (df['Grupo'].isin(selected_grupos)) &
                (df['Marca'].isin(selected_marcas)) &
                (df['Tipo_Proveedor'].isin(selected_proveedores))
            )
            
            # Aplicar filtro de SKU también al periodo anterior si aplica
            if sku_input.strip():
                skus_to_filter = [int(s.strip()) for s in re.split('[ ,\n]+', sku_input) if s.strip().isdigit()]
                if skus_to_filter:
                    base_filters = base_filters & (df['CodPro'].isin(skus_to_filter))

            df_previous = df[base_filters & (df['Fecha'] >= prev_start_date) & (df['Fecha'] <= prev_end_date)]

            # Función para calcular métricas de un dataframe
            def get_metrics(dataf):
                if dataf.empty:
                    return {"vn": 0, "ub": 0, "cost": 0, "cant": 0}
                return {
                    "vn": dataf['Venta Neta'].sum(),
                    "ub": dataf['Utilidad Bruta'].sum(),
                    "cost": dataf['Costo'].sum(),
                    "cant": dataf['Cantidad'].sum()
                }

            current_metrics = get_metrics(df_filtered)
            previous_metrics = get_metrics(df_previous)

            # Función para crear el texto del delta
            def create_delta_text(current_val, prev_val, is_percentage=False, is_currency=True):
                if df_previous.empty or prev_val == 0: return None # No mostrar nada si no hay datos previos
                
                delta_perc = ((current_val - prev_val) / prev_val) * 100
                prev_val_str = f"${prev_val:,.0f}" if is_currency else f"{prev_val:,.0f}"
                
                if is_percentage:
                    delta_abs = current_val - prev_val
                    return f"{delta_abs:.2f} p.p. (vs {prev_val:.2f}%)"
                
                return f"{delta_perc:.1f}% (vs {prev_val_str})"


            # Calcular KPIs compuestos
            margen_bruto_curr = (current_metrics["ub"] / current_metrics["vn"]) * 100 if current_metrics["vn"] != 0 else 0
            margen_bruto_prev = (previous_metrics["ub"] / previous_metrics["vn"]) * 100 if previous_metrics["vn"] != 0 else 0
            
            ticket_promedio_curr = current_metrics["vn"] / current_metrics["cant"] if current_metrics["cant"] != 0 else 0
            ticket_promedio_prev = previous_metrics["vn"] / previous_metrics["cant"] if previous_metrics["cant"] != 0 else 0

            costo_unitario_curr = current_metrics["cost"] / current_metrics["cant"] if current_metrics["cant"] != 0 else 0
            costo_unitario_prev = previous_metrics["cost"] / previous_metrics["cant"] if previous_metrics["cant"] != 0 else 0

            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                st.metric(label="Venta Neta Total", value=f"${current_metrics['vn']:,.0f}", delta=create_delta_text(current_metrics['vn'], previous_metrics['vn']))
            with col2:
                st.metric(label="Costo Total", value=f"${current_metrics['cost']:,.0f}", delta=create_delta_text(current_metrics['cost'], previous_metrics['cost']))
            with col3:
                st.metric(label="Utilidad Bruta Total", value=f"${current_metrics['ub']:,.0f}", delta=create_delta_text(current_metrics['ub'], previous_metrics['ub']))
            with col4:
                st.metric(label="Margen Bruto", value=f"{margen_bruto_curr:.2f}%", delta=create_delta_text(margen_bruto_curr, margen_bruto_prev, is_percentage=True))
            with col5:
                st.metric(label="Unidades Vendidas", value=f"{current_metrics['cant']:,.0f}", delta=create_delta_text(current_metrics['cant'], previous_metrics['cant'], is_currency=False))
            with col6:
                st.metric(label="Ticket Promedio", value=f"${ticket_promedio_curr:,.0f}", delta=create_delta_text(ticket_promedio_curr, ticket_promedio_prev))
            with col7:
                st.metric(label="Costo Promedio x Un", value=f"${costo_unitario_curr:,.0f}", delta=create_delta_text(costo_unitario_curr, costo_unitario_prev))

            if not df_previous.empty:
                st.caption(f"Comparando con el periodo: {prev_start_date.strftime('%Y-%m-%d')} al {prev_end_date.strftime('%Y-%m-%d')}")

            # --- GRÁFICO DE CASCADA (P&G) CON NUEVOS COLORES Y TEXTO ---
            st.markdown("##### Desglose de la Utilidad Bruta (P&G)")
            
            fig_waterfall = go.Figure(go.Waterfall(
                name = "P&G", orientation = "v",
                measure = ["absolute", "relative", "total"],
                x = ["Venta Neta", "Costo", "Utilidad Bruta"],
                text = [f"${current_metrics['vn']:,.0f}", f"-$ {current_metrics['cost']:,.0f}", f"${current_metrics['ub']:,.0f}"],
                textposition = 'auto',
                textfont = {"size": 16}, # Aumentar tamaño de fuente
                y = [current_metrics['vn'], -current_metrics['cost'], current_metrics['ub']],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
                decreasing = {"marker":{"color":"#5E88FC"}}, # Azul medio para costos
                increasing = {"marker":{"color":"#1C3FAA"}}, # Azul oscuro para ventas
                totals = {"marker":{"color":"#00227A"}}     # Azul más oscuro para totales
            ))

            fig_waterfall.update_layout(
                    title = "Estado de Resultados Simplificado",
                    showlegend = False
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)


            st.markdown("<hr>", unsafe_allow_html=True)

            # --- GRÁFICOS GENERALES ---
            st.markdown("### Análisis General y por Sucursal")
            col_graf1, col_graf2 = st.columns(2)

            with col_graf1:
                st.markdown("##### Evolución Mensual")
                df_time_series = df_filtered.set_index('Fecha').groupby(pd.Grouper(freq='M')).agg(
                    {'Venta Neta': 'sum', 'Utilidad Bruta': 'sum'}
                ).reset_index()
                df_time_series['Margen Bruto'] = (df_time_series['Utilidad Bruta'] / df_time_series['Venta Neta']) * 100
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=df_time_series['Fecha'], y=df_time_series['Venta Neta'], name='Venta Neta', marker_color='#0068c9'), secondary_y=False)
                fig.add_trace(go.Bar(x=df_time_series['Fecha'], y=df_time_series['Utilidad Bruta'], name='Utilidad Bruta', marker_color='#ff7f0e'), secondary_y=False)
                fig.add_trace(go.Scatter(x=df_time_series['Fecha'], y=df_time_series['Margen Bruto'], name='Margen Bruto (%)', mode='lines+markers', line=dict(color='green')), secondary_y=True)
                
                fig.update_layout(title_text='Ventas, Utilidad y Margen en el Tiempo', barmode='group')
                fig.update_yaxes(title_text="Monto ($)", secondary_y=False)
                fig.update_yaxes(title_text="Margen (%)", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_graf2:
                st.markdown("##### Distribución de Ventas por Sucursal")
                sales_by_sucursal = df_filtered.groupby('Sucursal', as_index=False)['Venta Neta'].sum().sort_values('Venta Neta', ascending=False)
                fig_sucursal = px.bar(
                    sales_by_sucursal, x='Sucursal', y='Venta Neta',
                    title='Venta Neta por Sucursal', labels={'Venta Neta': 'Venta Neta Total ($)', 'Sucursal': 'Sucursal'},
                    text='Venta Neta'
                )
                fig_sucursal.update_traces(marker_color='#0068c9', texttemplate='$%{text:,.0f}', textposition='outside')
                fig_sucursal.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig_sucursal, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # --- ANÁLISIS POR TIPO DE PROVEEDOR ---
            st.markdown("### Análisis por Tipo de Proveedor")
            col_proveedor_1, col_proveedor_2 = st.columns(2)
            
            with col_proveedor_1:
                st.markdown("##### Composición de Venta por Proveedor")
                df_proveedor = df_filtered.groupby('Tipo_Proveedor')[['Costo', 'Utilidad Bruta']].sum()
                df_proveedor['Venta Neta'] = df_proveedor['Costo'] + df_proveedor['Utilidad Bruta']
                df_proveedor = df_proveedor.sort_values(by='Venta Neta', ascending=False).reset_index()

                fig_proveedor_stack = go.Figure()
                fig_proveedor_stack.add_trace(go.Bar(
                    y=df_proveedor['Tipo_Proveedor'], x=df_proveedor['Costo'],
                    name='Costo', orientation='h', marker=dict(color='rgba(255, 127, 14, 0.6)')
                ))
                fig_proveedor_stack.add_trace(go.Bar(
                    y=df_proveedor['Tipo_Proveedor'], x=df_proveedor['Utilidad Bruta'],
                    name='Utilidad Bruta', orientation='h', marker=dict(color='rgba(0, 104, 201, 0.6)')
                ))
                
                fig_proveedor_stack.update_layout(
                    barmode='stack',
                    title='Venta Neta (Costo + Utilidad) por Tipo de Proveedor',
                    xaxis_title='Venta Neta Total ($)',
                    yaxis_title='Tipo de Proveedor',
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig_proveedor_stack, use_container_width=True)

            with col_proveedor_2:
                st.markdown("##### Distribución de Venta Neta")
                sales_by_proveedor = df_filtered.groupby('Tipo_Proveedor', as_index=False)['Venta Neta'].sum()
                fig_proveedor_pie = px.pie(
                    sales_by_proveedor,
                    values='Venta Neta',
                    names='Tipo_Proveedor',
                    title='Venta Neta por Tipo de Proveedor',
                    hole=0.4
                )
                fig_proveedor_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_proveedor_pie, use_container_width=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)

            # --- ANÁLISIS POR GRUPO DE PAGO ---
            st.markdown("### Análisis por Grupo de Pago")
            col_grupo1, col_grupo2 = st.columns(2)

            sales_by_grupo = df_filtered.groupby('Grupo', as_index=False)['Venta Neta'].sum().sort_values('Venta Neta', ascending=False)
            
            if len(sales_by_grupo) > 5:
                top_5_grupos = sales_by_grupo.head(5).copy()
                otros_sum = sales_by_grupo.iloc[5:]['Venta Neta'].sum()
                sales_by_grupo_pareto = pd.concat(
                    [top_5_grupos, pd.DataFrame([{'Grupo': 'Otros', 'Venta Neta': otros_sum}])],
                    ignore_index=True
                )
            else:
                sales_by_grupo_pareto = sales_by_grupo

            with col_grupo1:
                st.markdown("##### Venta Neta por Grupo")
                fig_grupo_bar = px.bar(
                    sales_by_grupo_pareto,
                    x='Grupo',
                    y='Venta Neta',
                    title='Venta Neta Total por Grupo',
                    labels={'Venta Neta': 'Venta Neta Total ($)', 'Grupo': 'Grupo'},
                    text='Venta Neta'
                )
                fig_grupo_bar.update_traces(marker_color='#0068c9', texttemplate='$%{text:,.0f}', textposition='outside')
                fig_grupo_bar.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig_grupo_bar, use_container_width=True)

            with col_grupo2:
                st.markdown("##### Distribución de Venta por Grupo")
                fig_grupo_pie = px.pie(
                    sales_by_grupo_pareto,
                    values='Venta Neta',
                    names='Grupo',
                    title='Distribución % de Venta por Grupo',
                    hole=0.4
                )
                fig_grupo_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_grupo_pie, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # --- ANÁLISIS DE CANALES DE PAGO Y PERFIL DE COMPRA ---
            st.markdown("### Análisis de Canales de Pago y Perfil de Compra")

            df_grupo_metrics = df_filtered.groupby('Grupo', as_index=False).agg(
                Venta_Neta_Total=('Venta Neta', 'sum'),
                Cantidad_Total=('Cantidad', 'sum')
            )
            df_grupo_metrics['Ticket Promedio'] = (
                df_grupo_metrics['Venta_Neta_Total']
                .div(df_grupo_metrics['Cantidad_Total'])
                .replace([float('inf'), -float('inf')], 0).fillna(0)
            )
            df_grupo_metrics = df_grupo_metrics.sort_values('Ticket Promedio', ascending=False)

            col_canal1, col_canal2 = st.columns([1, 1.2])

            with col_canal1:
                st.markdown("##### Ticket Promedio por Grupo de Pago")
                fig_ticket_promedio = px.bar(
                    df_grupo_metrics,
                    x='Grupo',
                    y='Ticket Promedio',
                    title='Valor de Compra Promedio por Canal',
                    labels={'Ticket Promedio': 'Ticket Promedio ($)', 'Grupo': 'Grupo de Pago'},
                    text='Ticket Promedio'
                )
                fig_ticket_promedio.update_traces(
                    marker_color='#2ca02c', # Un color diferente para este análisis
                    texttemplate='$%{text:,.0f}',
                    textposition='outside'
                )
                fig_ticket_promedio.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig_ticket_promedio, use_container_width=True)

            with col_canal2:
                # --- CAMBIO: Se elimina el filtro selectbox ---
                st.markdown("##### Composición de Marcas en Grupos Seleccionados")
                
                df_treemap_data = df_filtered.groupby('Marca', as_index=False)['Venta Neta'].sum()
                
                if not df_treemap_data.empty:
                    fig_treemap = px.treemap(
                        df_treemap_data,
                        path=[px.Constant('Todas las Marcas'), 'Marca'],
                        values='Venta Neta',
                        title='Distribución de Ventas por Marca',
                        color='Venta Neta',
                        color_continuous_scale='Greens'
                    )
                    fig_treemap.update_layout(margin = dict(t=50, l=25, r=25, b=25))
                    st.plotly_chart(fig_treemap, use_container_width=True)
                else:
                    st.info("No hay datos de marcas para los filtros seleccionados.")

            st.markdown("<hr>", unsafe_allow_html=True)
            
            # --- MATRIZ DE RENDIMIENTO GEOGRÁFICO ---
            st.markdown("### Matriz de Rendimiento Geográfico (Sucursal vs. Marca)")
            
            heatmap_metric = st.selectbox(
                "Selecciona la métrica para el mapa de calor:",
                options=['Venta Neta', 'Utilidad Bruta', 'Margen Bruto (%)'],
                key='heatmap_metric'
            )

            if heatmap_metric == 'Margen Bruto (%)':
                pivot_data = df_filtered.groupby(['Sucursal', 'Marca'], as_index=False).agg(
                    Venta_Neta_Sum=('Venta Neta', 'sum'),
                    Utilidad_Bruta_Sum=('Utilidad Bruta', 'sum')
                )
                pivot_data['Margen Bruto (%)'] = (
                    pivot_data['Utilidad_Bruta_Sum']
                    .div(pivot_data['Venta_Neta_Sum'])
                    .mul(100)
                    .replace([float('inf'), -float('inf')], 0).fillna(0)
                )
                pivot_table = pivot_data.pivot_table(
                    index='Sucursal', columns='Marca', values='Margen Bruto (%)', aggfunc='mean'
                )
                text_format = ".1f"
                color_scale = "RdYlGn"
            else: # Venta Neta o Utilidad Bruta
                pivot_table = df_filtered.pivot_table(
                    index='Sucursal', columns='Marca', values=heatmap_metric, aggfunc='sum'
                )
                text_format = ",.0f"
                color_scale = "Blues" if heatmap_metric == 'Venta Neta' else "Oranges"

            pivot_table.fillna(0, inplace=True)

            if not pivot_table.empty:
                fig_heatmap = px.imshow(
                    pivot_table,
                    text_auto=True,
                    aspect="auto",
                    labels=dict(x="Marca", y="Sucursal", color=f"{heatmap_metric}"),
                    title=f"Rendimiento de {heatmap_metric} por Sucursal y Marca",
                    color_continuous_scale=color_scale
                )
                fig_heatmap.update_traces(texttemplate=f"%{{z:{text_format}}}")
                fig_heatmap.update_xaxes(side="top")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No hay suficientes datos para generar la matriz de rendimiento con los filtros actuales.")

            st.markdown("<hr>", unsafe_allow_html=True)
            
            # --- ANÁLISIS DE VENTAS POR PRODUCTO Y MARCA ---
            st.markdown("### Análisis de Ventas por Producto y Marca")
            col_graf3, col_graf4 = st.columns(2)

            with col_graf3:
                st.markdown("##### Top 10 Marcas por Venta Neta")
                sales_by_marca = df_filtered.groupby('Marca', as_index=False)['Venta Neta'].sum().nlargest(10, 'Venta Neta').sort_values('Venta Neta', ascending=True)
                fig_marca = px.bar(
                    sales_by_marca,
                    x='Venta Neta',
                    y='Marca',
                    orientation='h',
                    title='Top 10 Marcas por Venta Neta',
                    labels={'Venta Neta': 'Venta Neta Total ($)', 'Marca': 'Marca'},
                    text='Venta Neta'
                )
                fig_marca.update_traces(texttemplate='$%{text:,.0f}', textposition='outside', marker_color='#0068c9')
                fig_marca.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig_marca, use_container_width=True)

            with col_graf4:
                st.markdown("##### Top 10 Productos (Venta y Unidades)")
                top_products = df_filtered.groupby('Producto').agg({'Venta Neta': 'sum', 'Cantidad': 'sum'}).nlargest(10, 'Venta Neta').sort_values('Venta Neta', ascending=True).reset_index()
                
                fig_top_prod = go.Figure()
                fig_top_prod.add_trace(go.Bar(y=top_products['Producto'], x=top_products['Venta Neta'], name='Venta Neta', orientation='h', marker_color='#0068c9'))
                # --- CAMBIO: Quitar la línea de conexión, dejar solo marcadores ---
                fig_top_prod.add_trace(go.Scatter(
                    y=top_products['Producto'], 
                    x=top_products['Cantidad'], 
                    name='Unidades', 
                    mode='markers', # Solo marcadores
                    marker=dict(color='darkred', size=8), # Aumentar tamaño del marcador
                    xaxis='x2'
                ))
                
                fig_top_prod.update_layout(
                    title_text='Top 10 Productos más Vendidos',
                    xaxis=dict(title='Venta Neta Total ($)'),
                    xaxis2=dict(title='Unidades Vendidas', overlaying='x', side='top'),
                    yaxis=dict(title='Producto'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_top_prod, use_container_width=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # --- ANÁLISIS DE RENTABILIDAD ---
            st.markdown("### Análisis de Rentabilidad")
            col_graf5, col_graf6 = st.columns(2)

            with col_graf5:
                st.markdown("##### Top 10 Marcas por Utilidad Bruta")
                profit_by_marca = df_filtered.groupby('Marca', as_index=False)['Utilidad Bruta'].sum().nlargest(10, 'Utilidad Bruta').sort_values('Utilidad Bruta', ascending=True)
                fig_profit_marca = px.bar(
                    profit_by_marca,
                    x='Utilidad Bruta',
                    y='Marca',
                    orientation='h',
                    title='Top 10 Marcas por Utilidad Bruta',
                    labels={'Utilidad Bruta': 'Utilidad Bruta Total ($)', 'Marca': 'Marca'},
                    text='Utilidad Bruta'
                )
                fig_profit_marca.update_traces(texttemplate='$%{text:,.0f}', textposition='outside', marker_color='#ff7f0e')
                fig_profit_marca.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig_profit_marca, use_container_width=True)

            with col_graf6:
                st.markdown("##### Top 10 Productos más Rentables (por Utilidad Bruta)")
                top_profitable_products = df_filtered.groupby('Producto', as_index=False)['Utilidad Bruta'].sum().nlargest(10, 'Utilidad Bruta').sort_values('Utilidad Bruta', ascending=True)
                fig_top_profit_prod = px.bar(
                    top_profitable_products, x='Utilidad Bruta', y='Producto', orientation='h',
                    title='Top 10 Productos por Utilidad', labels={'Utilidad Bruta': 'Utilidad Bruta Total ($)', 'Producto': 'Producto'},
                    text='Utilidad Bruta'
                )
                fig_top_profit_prod.update_traces(texttemplate='$%{text:,.0f}', textposition='outside', marker_color='#ff7f0e')
                fig_top_profit_prod.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig_top_profit_prod, use_container_width=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)

            # --- ANÁLISIS DE ELASTICIDAD POR PRODUCTO ---
            st.markdown("### Análisis de Elasticidad por Producto")
            
            # --- CAMBIO: Se elimina el filtro y se calcula el producto más vendido ---
            top_product_series = df_filtered.groupby('Producto')['Venta Neta'].sum().nlargest(1)
            
            if not top_product_series.empty:
                producto_seleccionado = top_product_series.index[0]
                
                # Filtrar datos para el producto seleccionado
                df_producto = df_filtered[df_filtered['Producto'] == producto_seleccionado].copy()
                
                # Agrupar por mes y calcular métricas
                df_elasticidad = df_producto.set_index('Fecha').groupby(pd.Grouper(freq='M')).agg(
                    Precio_Promedio=('Precio/Unidad', 'mean'),
                    Costo_Promedio=('Costo/Unidad', 'mean'),
                    Cantidad_Total=('Cantidad', 'sum')
                ).reset_index()

                if not df_elasticidad.empty and len(df_elasticidad) > 1:
                    # Crear el gráfico combinado con doble eje
                    fig_elasticidad = make_subplots(specs=[[{"secondary_y": True}]])

                    # Eje 1: Barras para la cantidad
                    fig_elasticidad.add_trace(
                        go.Bar(
                            x=df_elasticidad['Fecha'], 
                            y=df_elasticidad['Cantidad_Total'], 
                            name='Cantidad Vendida', 
                            marker_color='rgba(0, 104, 201, 0.6)'
                        ), 
                        secondary_y=False
                    )

                    # Eje 2: Líneas para precio y costo
                    fig_elasticidad.add_trace(
                        go.Scatter(
                            x=df_elasticidad['Fecha'], 
                            y=df_elasticidad['Precio_Promedio'], 
                            name='Precio Prom./Unidad', 
                            mode='lines+markers',
                            line=dict(color='green')
                        ), 
                        secondary_y=True
                    )
                    fig_elasticidad.add_trace(
                        go.Scatter(
                            x=df_elasticidad['Fecha'], 
                            y=df_elasticidad['Costo_Promedio'], 
                            name='Costo Prom./Unidad', 
                            mode='lines+markers',
                            line=dict(color='red')
                        ), 
                        secondary_y=True
                    )

                    # Configurar el layout del gráfico
                    fig_elasticidad.update_layout(
                        title_text=f'Dinámica de Precio y Volumen para el producto: <b>{producto_seleccionado}</b>',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig_elasticidad.update_xaxes(title_text="Mes")
                    fig_elasticidad.update_yaxes(title_text="Cantidad Vendida (Unidades)", secondary_y=False)
                    fig_elasticidad.update_yaxes(title_text="Monto Promedio ($)", secondary_y=True)
                    
                    st.plotly_chart(fig_elasticidad, use_container_width=True)
                else:
                    st.info(f"No hay suficientes datos mensuales para analizar la elasticidad del producto '{producto_seleccionado}'.")
            else:
                st.info("No se encontró un producto más vendido con los filtros actuales.")


            st.markdown("<hr>", unsafe_allow_html=True)

            # --- TABLA DE DATOS DETALLADA CON PAGINACIÓN ---
            st.markdown("## 📄 Datos Filtrados")
            st.markdown("Explora los datos detallados según los filtros aplicados.")
            
            rows_per_page = 50
            total_rows = len(df_filtered)
            total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page > 0 else 0)
            page_number = 1
            if total_pages > 1:
                page_number = st.number_input(f'Página (1-{total_pages})', min_value=1, max_value=total_pages, value=1, step=1)
            
            start_idx = (page_number - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            df_to_display = df_filtered.iloc[start_idx:end_idx].copy()
            
            st.caption(f"Mostrando filas {start_idx+1} a {min(end_idx, total_rows)} de {total_rows}")

            # Se usa la columna 'Periodo' para la tabla
            display_cols = [
                'Periodo', 'Sucursal', 'Producto', 'Marca', 'Cantidad', 
                'Venta Neta', 'Precio/Unidad', 'Costo', 'Costo/Unidad', 
                'Utilidad Bruta', 'Margen Utilidad (%)'
            ]

            st.dataframe(df_to_display[display_cols].style.format({
                'Venta Neta': '${:,.0f}', 'Precio/Unidad': '${:,.0f}',
                'Costo': '${:,.0f}', 'Costo/Unidad': '${:,.0f}',
                'Utilidad Bruta': '${:,.0f}', 'Margen Utilidad (%)': '{:.2f}%'
            }), use_container_width=True)

            # --- BOTÓN DE DESCARGA ---
            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                # Se usan las columnas de display que ya tienen 'Periodo'
                return df_to_convert[display_cols].to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(df_filtered)
            st.download_button(
               label="📥 Descargar datos filtrados como CSV",
               data=csv, file_name='datos_filtrados.csv', mime='text/csv', key='download_main'
            )
    elif df is None:
        st.error("No se pudo cargar el archivo. Revisa el mensaje de error en la parte superior.")
    else: # df is empty after cleaning
        st.error("No se encontraron datos válidos en el archivo cargado después de la limpieza.")
