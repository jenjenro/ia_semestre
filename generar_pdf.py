from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Documentacion del Dashboard (Explicacion Linea a Linea)', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

pdf = PDF()
pdf.add_page()
pdf.set_font('helvetica', '', 11)

texto = """ESTRUCTURA DEL DASHBOARD (FRONTEND)

El dashboard se compone de dos archivos principales: index.html y app.js.

1. index.html (Estructura y Estilos)
- Usamos Tailwind CSS para dar estilo sin escribir CSS manual.
- FontAwesome provee los iconos de interfaz.
- La interfaz se divide en dos columnas principales:
  * Panel Izquierdo: Contiene el formulario de subida de archivos (uploadForm).
  * Panel Derecho: Muestra el historial de archivos procesados.
- Debajo del formulario hay una seccion oculta (resultsContent) que se vuelve visible una vez que la API responde.

2. app.js (Logica e Interactividad)
- Linea 1: const API_BASE ...
  Detecta dinamicamente en que puerto estas ejecutando el dashboard para evitar errores CORS.

- Variables (Lineas 3-14): Se obtienen las referencias a los elementos HTML usando document.getElementById() para manipularlos.

- EVENTO SUBMIT (Linea 17): uploadForm.addEventListener...
  Captura el momento en que haces clic en Procesar. preventDefault() evita que la pagina recargue. Luego, envia el archivo via fetch() a /api/upload.
  
- RENDERIZAR RESULTADOS: Dependiendo de si la API responde con image o document, llama a renderImageResults o renderTextResults.

- HISTORIAL Y CLICK:
  - loadHistory hace una peticion GET a /api/history y genera divs HTML.
  - A cada div se le anade la funcion onclick=handleHistoryClick(this).
  - handleHistoryClick: Lee la info guardada y llama a renderHistoryResult().

- ELIMINAR HISTORIAL:
  - La peticion a /api/history vacia toda la tabla.
  - La peticion a /api/history/{id} borra un registro individual.
"""

for line in texto.split('\n'):
    pdf.multi_cell(0, 7, text=line)

pdf.output('explicacion_dashboard.pdf')
print('PDF generado exitosamente')
