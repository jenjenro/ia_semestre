from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def crear_pdf():
    doc = SimpleDocTemplate("Explicacion_Dashboard.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = styles['Title']
    h1_style = styles['Heading1']
    h2_style = styles['Heading2']
    normal_style = styles['Normal']
    code_style = ParagraphStyle('Code', parent=normal_style, fontName='Courier', fontSize=9, backColor='#f0f0f0', borderPadding=5)
    
    flowables = []
    
    flowables.append(Paragraph("Explicacion Linea a Linea del Dashboard", title_style))
    flowables.append(Spacer(1, 12))
    
    flowables.append(Paragraph("El dashboard se compone de dos archivos principales en la carpeta dashboard/: <b>index.html</b> y <b>app.js</b>.", normal_style))
    flowables.append(Spacer(1, 12))
    
    # Section 1
    flowables.append(Paragraph("1. index.html (Estructura y Estilos)", h1_style))
    flowables.append(Paragraph("<b>Tailwind CSS y FontAwesome:</b> En el head cargamos Tailwind via CDN para dar estilos rapidamente sin CSS adicional. Tambien se cargan los iconos de FontAwesome.", normal_style))
    flowables.append(Spacer(1, 6))
    flowables.append(Paragraph("<b>Estructura de Columnas:</b> La pagina esta dividida. El <b>Panel Izquierdo</b> contiene el formulario de subida de archivos (uploadForm). El <b>Panel Derecho</b> muestra el Historial, el boton de actualizar y el boton de vaciar historial.", normal_style))
    flowables.append(Spacer(1, 6))
    flowables.append(Paragraph("<b>Resultados y Modal:</b> Debajo del formulario hay una seccion oculta que se hace visible tras la respuesta de la API. Ademas hay un div de imagen en pantalla completa (imageModal).", normal_style))
    flowables.append(Spacer(1, 12))
    
    # Section 2
    flowables.append(Paragraph("2. app.js (Logica e Interactividad)", h1_style))
    flowables.append(Paragraph("Este archivo envia y recibe datos del servidor FastAPI en Python.", normal_style))
    flowables.append(Spacer(1, 12))
    
    flowables.append(Paragraph("<b>Configuracion de Red:</b>", h2_style))
    flowables.append(Paragraph("const API_BASE = (window.location.port === '8000') ? '' : `http://${window.location.hostname || 'localhost'}:8000`;", code_style))
    flowables.append(Paragraph("Calcula dinamicamente la URL del servidor para evitar problemas de CORS.", normal_style))
    flowables.append(Spacer(1, 12))
    
    flowables.append(Paragraph("<b>Subida de Archivos (Linea 17):</b>", h2_style))
    flowables.append(Paragraph("uploadForm.addEventListener('submit', async (e) => { ... })", code_style))
    flowables.append(Paragraph("Captura el clic en Procesar, evita que recargue la pagina (e.preventDefault) y envia la peticion POST a /api/upload usando fetch().", normal_style))
    flowables.append(Spacer(1, 12))
    
    flowables.append(Paragraph("<b>Renderizado Dinamico:</b>", h2_style))
    flowables.append(Paragraph("Dependiendo de si data.type es 'image' o 'document', llama a renderImageResults o renderTextResults para armar el HTML de las capas visuales o las barras de probabilidad en tiempo real.", normal_style))
    flowables.append(Spacer(1, 12))
    
    flowables.append(Paragraph("<b>Historial (GET /api/history):</b>", h2_style))
    flowables.append(Paragraph("La funcion loadHistory() obtiene los datos de SQLite y arma la lista visual. Al hacer clic, handleHistoryClick() lee el atributo 'data-item' escondido en cada recuadro y repinta los resultados en pantalla.", normal_style))
    flowables.append(Spacer(1, 12))
    
    flowables.append(Paragraph("<b>Eliminacion (DELETE /api/history):</b>", h2_style))
    flowables.append(Paragraph("clearHistoryBtn vacia toda la tabla en el servidor, mientras que deleteHistoryItem() manda el ID especifico del documento para borrar uno solo.", normal_style))
    
    doc.build(flowables)

if __name__ == "__main__":
    crear_pdf()

