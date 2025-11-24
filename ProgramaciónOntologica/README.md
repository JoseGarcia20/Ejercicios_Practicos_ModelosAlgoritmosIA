Ontología IoT para Hogar Inteligente (Python + SPARQL)

Descripción
- Construye una ontología para IoT en un hogar inteligente usando RDF/OWL con `rdflib`.
- Modela clases principales: `Dispositivo`, `Sensor`, `Habitacion`, `Automatizacion` y relaciones: `ubicadoEn`, `controla`, `detecta`.
- Axioma implementado (razonamiento simple): “los dispositivos de la misma habitación comparten contexto” → se materializa con la propiedad simétrica `comparteContextoCon` entre dispositivos que comparten `ubicadoEn`.
- Incluye un caso real con habitaciones, dispositivos, sensores y automatizaciones, más un flujo de simulación básico de eventos y acciones.
- Permite consultas SPARQL interactivas.

Requisitos
- Python 3.9+
- Paquetes: `rdflib`

Instalación
1) Crear entorno e instalar dependencias:
   - `python -m venv .venv`
   - Windows: `.venv\\Scripts\\activate`
   - `pip install -r requirements.txt`

Ejecución
- Modo demo (construye ontología, corre razonador, ejecuta simulación y muestra consultas ejemplo):
  - `python main.py --demo`

- Consola SPARQL interactiva sobre el grafo construido:
  - `python main.py --repl`
  - Escriba `help` para ver ejemplos.
  - Para salir: `exit` o `quit`.

Arquitectura
- `src/ontologia_iot/ontology.py`: Define namespaces, clases, propiedades y puebla el caso real.
- `src/ontologia_iot/reasoner.py`: Razonamiento simple para el axioma de contexto compartido.
- `src/ontologia_iot/queries.py`: Consultas SPARQL de ejemplo.
- `main.py`: CLI para demo y REPL SPARQL.

Ejemplos de consultas SPARQL
- Dispositivos por habitación:
  - `PREFIX iot: <http://example.org/iot#>`
  - `SELECT ?hab ?dis WHERE { ?dis a iot:Dispositivo ; iot:ubicadoEn ?hab . }`

- Sensores y lo que detectan:
  - `PREFIX iot: <http://example.org/iot#>`
  - `SELECT ?sensor ?evento WHERE { ?sensor a iot:Sensor ; iot:detecta ?evento . }`

- Automatizaciones y dispositivos controlados:
  - `PREFIX iot: <http://example.org/iot#>`
  - `SELECT ?auto ?disp WHERE { ?auto a iot:Automatizacion ; iot:controla ?disp . }`

- Dispositivos que comparten contexto (axioma materializado):
  - `PREFIX iot: <http://example.org/iot#>`
  - `SELECT ?a ?b WHERE { ?a a iot:Dispositivo ; iot:comparteContextoCon ?b . FILTER(?a != ?b) }`

