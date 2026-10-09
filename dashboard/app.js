const API_BASE = (window.location.port === '8000') ? '' : `http://${window.location.hostname || 'localhost'}:8000`;

const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const submitBtn = document.getElementById('submitBtn');
const resetBtn = document.getElementById('resetBtn');
const loadingState = document.getElementById('loadingState');
const emptyState = document.getElementById('emptyState');
const resultsContent = document.getElementById('resultsContent');
const historyList = document.getElementById('historyList');
const refreshHistoryBtn = document.getElementById('refreshHistory');
const imageModal = document.getElementById('imageModal');
const modalImg = document.getElementById('modalImg');
const closeModalBtn = document.getElementById('closeModal');

// Subida de Archivo
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files || fileInput.files.length === 0) return;

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    // Update UI state
    submitBtn.disabled = true;
    submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
    emptyState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    loadingState.classList.remove('hidden');
    resultsContent.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || data.detail || `Error del servidor: ${response.status}`);
        }

        renderResults(data);
        loadHistory();
        
        // Show reset button
        submitBtn.classList.add('hidden');
        resetBtn.classList.remove('hidden');

    } catch (error) {
        resultsContent.innerHTML = `
            <div class="bg-red-50 text-red-600 p-4 rounded-lg border border-red-200">
                <p class="font-bold"><i class="fas fa-exclamation-triangle"></i> Error al procesar</p>
                <p class="text-sm mt-1">${error.message}</p>
            </div>
        `;
        resultsContent.classList.remove('hidden');
    } finally {
        loadingState.classList.add('hidden');
        submitBtn.disabled = false;
        submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
});

// Resetear formulario
resetBtn.addEventListener('click', () => {
    uploadForm.reset();
    submitBtn.classList.remove('hidden');
    resetBtn.classList.add('hidden');
    resultsContent.classList.add('hidden');
    emptyState.classList.remove('hidden');
    resultsContent.innerHTML = '';
});

// Renderizar Resultados
function renderResults(data) {
    if (data.type === 'image') {
        renderImageResults(data);
    } else {
        renderTextResults(data);
    }
    resultsContent.classList.remove('hidden');
}

function renderImageResults(data) {
    const { details, images } = data;
    const cat = details.categoria || '';
    const exp = details.explicacion || '';
    
    resultsContent.innerHTML = `
        <!-- Main Result Card -->
        <div class="bg-indigo-50 border border-indigo-100 rounded-lg p-5">
            <div class="text-indigo-600 font-bold text-sm uppercase tracking-wide mb-1">Diagnóstico (Visión Computacional)</div>
            <div class="text-2xl font-bold text-gray-800 mb-3">${cat}</div>
            
            <div class="bg-white rounded border border-indigo-100 p-3 mb-3">
                <div class="text-xs font-bold text-indigo-500 uppercase mb-1">Ruta de Enrutamiento</div>
                <div class="text-sm">De <strong>${details.dominio}</strong> &rarr; a <strong>${details.destino}</strong></div>
            </div>

            <div class="bg-white rounded border border-pink-200 p-4 shadow-sm">
                <div class="text-xs font-bold text-pink-600 uppercase mb-3"><i class="fas fa-cogs"></i> ¿Cómo se clasificó esta imagen?</div>
                <ul class="text-sm text-gray-700 space-y-2 mb-2">
                    <li class="flex items-start">
                        <i class="fas fa-check-circle text-green-500 mt-1 mr-2"></i>
                        <span><strong>1. Detección de Bordes (Canny):</strong> Se identificaron las formas geométricas en la matriz de píxeles.</span>
                    </li>
                    <li class="flex items-start">
                        <i class="fas fa-check-circle text-green-500 mt-1 mr-2"></i>
                        <span><strong>2. Segmentación Automática (Otsu):</strong> El sistema separó las líneas estructurales del fondo.</span>
                    </li>
                    <li class="flex items-start">
                        <i class="fas fa-check-circle text-green-500 mt-1 mr-2"></i>
                        <span><strong>3. Extracción de Rasgos Morfológicos:</strong> ${exp}</span>
                    </li>
                </ul>
            </div>
        </div>

        <!-- Vision Grid -->
        <div class="mt-2">
            <h3 class="font-bold text-gray-700 mb-3 border-b pb-2">Capas de Procesamiento Visual</h3>
            <div class="grid grid-cols-2 gap-4">
                ${createImageCard('1. Original', images.original)}
                ${createImageCard('2. Contornos Canny', images.edges)}
                ${createImageCard('3. Máscara Otsu', images.mask)}
                ${createImageCard('4. Regiones Conectadas', images.regions)}
            </div>
        </div>
    `;

    attachImageModals();
}

function renderTextResults(data) {
    const { details } = data;
    const exp = details.explicacion || '';

    // Probabilities bars
    let probsHtml = '';
    if (details.probabilidades) {
        for (const [cls, prob] of Object.entries(details.probabilidades)) {
            const perc = (prob * 100).toFixed(1);
            probsHtml += `
                <div class="flex items-center text-sm mb-1">
                    <span class="w-32 truncate text-gray-600">${cls}</span>
                    <div class="flex-grow bg-gray-200 rounded-full h-2 mx-2">
                        <div class="bg-indigo-500 h-2 rounded-full" style="width: ${perc}%"></div>
                    </div>
                    <span class="w-12 text-right font-mono text-xs text-gray-500">${perc}%</span>
                </div>
            `;
        }
    }

    resultsContent.innerHTML = `
        <div class="bg-indigo-50 border border-indigo-100 rounded-lg p-5">
            <div class="text-indigo-600 font-bold text-sm uppercase tracking-wide mb-1">Diagnóstico (Red Neuronal MLP)</div>
            <div class="text-2xl font-bold text-gray-800 mb-3">${details.clase.toUpperCase()}</div>
            
            <div class="bg-white rounded border border-indigo-100 p-3 mb-3">
                <div class="text-xs font-bold text-indigo-500 uppercase mb-1">Análisis Semántico</div>
                <div class="text-sm">${details.interpretacion}</div>
            </div>

            <div class="bg-white rounded border border-pink-200 p-4 mb-4 shadow-sm">
                <div class="text-xs font-bold text-pink-600 uppercase mb-3"><i class="fas fa-cogs"></i> ¿Cómo se clasificó este documento?</div>
                
                <ul class="text-sm text-gray-700 space-y-2 mb-3">
                    <li class="flex items-start">
                        <i class="fas fa-check-circle text-green-500 mt-1 mr-2"></i>
                        <span><strong>1. Extracción de Texto:</strong> Se extrajeron exitosamente ${data.metrics.Caracteres} caracteres legibles del documento original.</span>
                    </li>
                    <li class="flex items-start">
                        <i class="fas fa-check-circle text-green-500 mt-1 mr-2"></i>
                        <span><strong>2. Vectorización (TF-IDF):</strong> El texto crudo fue transformado matemáticamente. Los términos más relevantes encontrados fueron: <span class="bg-pink-100 text-pink-800 px-1 py-0.5 rounded font-mono text-xs">${exp.replace("El modelo encontró términos clave (", "").replace(") que activaron fuertemente la categoría '" + details.clase.toUpperCase() + "' en la Red Neuronal Multicapa (MLP).", "").replace("No se encontraron términos altamente predictivos en el texto extraído.", "Ninguno destacado")}</span>.</span>
                    </li>
                    <li class="flex items-start">
                        <i class="fas fa-check-circle text-green-500 mt-1 mr-2"></i>
                        <span><strong>3. Red Neuronal (MLP):</strong> Las capas ocultas de la IA evaluaron estos términos y generaron una distribución probabilística, confirmando la clase final con un ${data.metrics.Confianza} de certeza.</span>
                    </li>
                </ul>
            </div>

            <div class="bg-white rounded border border-indigo-100 p-4">
                <div class="text-xs font-bold text-gray-500 uppercase mb-2">Distribución de Probabilidades Generada por la Red</div>
                ${probsHtml}
            </div>
        </div>

        <div class="mt-2">
            <h3 class="font-bold text-gray-700 mb-3 border-b pb-2">Texto Extraído Originalmente</h3>
            <div class="bg-gray-50 border border-gray-200 rounded p-4 text-xs font-mono text-gray-600 h-64 overflow-y-auto whitespace-pre-wrap">
                ${details.texto}
            </div>
        </div>
    `;
}

function createImageCard(title, base64) {
    return `
        <div class="border border-gray-200 rounded overflow-hidden bg-gray-50">
            <div class="text-xs font-bold text-center bg-gray-200 py-1 text-gray-600">${title}</div>
            <img src="data:image/png;base64,${base64}" class="w-full h-40 object-cover cursor-pointer hover:opacity-90 transition zoom-img">
        </div>
    `;
}

const clearHistoryBtn = document.getElementById('clearHistory');

// Historial
async function loadHistory() {
    try {
        const res = await fetch(`${API_BASE}/api/history`);
        const data = await res.json();
        
        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => {
                const cat = (item.categoria_predicha || item.clase_predicha || 'Desconocido').split(' ')[0];
                const file = item.fuente_documento || 'documento';
                const date = item.timestamp ? new Date(item.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : '';
                // Store item data as JSON string in data attribute
                const itemDataStr = escapeHtml(JSON.stringify(item));
                return `
                    <div class="bg-gray-50 border border-gray-200 rounded p-3 text-sm cursor-pointer hover:bg-indigo-50 transition-colors group relative" onclick="handleHistoryClick(this)" data-item="${itemDataStr}">
                        <div class="flex justify-between items-center mb-1">
                            <span class="font-bold text-indigo-700">${cat.toUpperCase()}</span>
                            <span class="text-xs text-gray-400">${date}</span>
                        </div>
                        <div class="truncate text-gray-600 pr-6" title="${file}">${file}</div>
                        <button onclick="deleteHistoryItem(event, ${item.id})" class="absolute right-3 bottom-3 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity" title="Eliminar documento">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                `;
            }).join('');
        } else {
            historyList.innerHTML = '<div class="text-sm text-gray-500 text-center mt-4">No hay documentos procesados aún.</div>';
        }
    } catch (e) {
        historyList.innerHTML = '<div class="text-sm text-red-500 text-center mt-4">Error cargando historial.</div>';
    }
}

refreshHistoryBtn.addEventListener('click', async () => {
    const icon = refreshHistoryBtn.querySelector('i');
    if (icon) icon.classList.add('fa-spin');
    await loadHistory();
    if (icon) setTimeout(() => icon.classList.remove('fa-spin'), 500);
});

if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', async () => {
        if (!confirm('¿Estás seguro de que quieres borrar TODO el historial?')) return;
        try {
            await fetch(`${API_BASE}/api/history`, { method: 'DELETE' });
            await loadHistory();
            resultsContent.innerHTML = '';
            resultsContent.classList.add('hidden');
            emptyState.classList.remove('hidden');
        } catch (e) {
            console.error(e);
        }
    });
}

window.deleteHistoryItem = async function(e, id) {
    e.stopPropagation(); // Evitar que se dispare el click del item
    if (!confirm('¿Eliminar este documento del historial?')) return;
    try {
        await fetch(`${API_BASE}/api/history/${id}`, { method: 'DELETE' });
        await loadHistory();
    } catch (err) {
        console.error(err);
    }
};

loadHistory();

// Image Modal Logic
function attachImageModals() {
    document.querySelectorAll('.zoom-img').forEach(img => {
        img.addEventListener('click', () => {
            modalImg.src = img.src;
            imageModal.classList.remove('hidden');
        });
    });
}
closeModalBtn.addEventListener('click', () => imageModal.classList.add('hidden'));
imageModal.addEventListener('click', (e) => {
    if (e.target === imageModal) imageModal.classList.add('hidden');
});

// Utilities
function escapeHtml(text) {
    if (!text) return '';
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return String(text).replace(/[&<>"']/g, function(m) { return map[m]; });
}

// Handle History Click
window.handleHistoryClick = function(element) {
    try {
        const itemData = JSON.parse(element.getAttribute('data-item'));
        renderHistoryResult(itemData);
    } catch (e) {
        console.error("Error parsing history item data:", e);
    }
};

function renderHistoryResult(item) {
    // Hide empty state and show results
    emptyState.classList.add('hidden');
    resultsContent.classList.remove('hidden');
    resetBtn.classList.remove('hidden');
    submitBtn.classList.add('hidden');

    const cat = item.categoria_predicha || 'Desconocido';
    const conf = typeof item.nivel_confianza === 'number' ? (item.nivel_confianza * 100).toFixed(1) + '%' : 'N/A';
    const filename = item.fuente_documento || 'Documento histórico';
    const semantic = item.significado_ontologico || 'Sin interpretación detallada';
    const features = item.resumen_contenido || 'Sin características extraídas';

    resultsContent.innerHTML = `
        <div class="bg-indigo-50 border border-indigo-100 rounded-lg p-5">
            <div class="flex justify-between items-start mb-2">
                <div>
                    <div class="text-indigo-600 font-bold text-sm uppercase tracking-wide mb-1">Diagnóstico Histórico</div>
                    <div class="text-2xl font-bold text-gray-800">${cat.toUpperCase()}</div>
                </div>
                <div class="bg-indigo-100 text-indigo-800 text-xs font-bold px-2 py-1 rounded">Confianza: ${conf}</div>
            </div>
            
            <div class="text-sm text-gray-500 mb-3"><i class="fas fa-file-alt mr-1"></i> ${escapeHtml(filename)}</div>
            
            <div class="bg-white rounded border border-indigo-100 p-3 mb-3">
                <div class="text-xs font-bold text-indigo-500 uppercase mb-1">Análisis Semántico / Enrutamiento</div>
                <div class="text-sm text-gray-700">${escapeHtml(semantic)}</div>
            </div>

            <div class="bg-white rounded border border-pink-200 p-4 shadow-sm">
                <div class="text-xs font-bold text-pink-600 uppercase mb-3"><i class="fas fa-search-plus"></i> ¿Por qué se clasificó así? (Características Identificadas)</div>
                <p class="text-sm text-gray-700 mb-2">
                    El sistema identificó las siguientes características intrínsecas en el documento original que llevaron a clasificarlo como <strong>${escapeHtml(cat)}</strong>:
                </p>
                <div class="bg-gray-50 border border-gray-200 rounded p-3 text-sm font-mono text-gray-600 mb-3 break-words whitespace-pre-wrap">
                    ${escapeHtml(features)}
                </div>
                <p class="text-xs text-gray-500">
                    <i class="fas fa-info-circle mr-1"></i> Dependiendo del tipo de documento, estas características pueden ser términos clave extraídos del texto (TF-IDF) o características visuales de la imagen (intensidad, bordes, regiones).
                </p>
            </div>
        </div>
    `;
}

