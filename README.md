# PAIR — Personal AI Router

PAIR es un proxy local que decide, en cada petición, si la resuelve tu modelo local (LM Studio) o un modelo de frontera (OpenAI). Lo sencillo se queda en tu equipo y no cuesta nada; lo difícil va a la API de pago.

Esta es la **fase 0 (prueba de concepto)**: proxy en Python, reglas en YAML y registro en SQLite. La app de escritorio llega en la fase 1.

## Qué hace

- **Tres puertas de entrada**, para que tus clientes funcionen sin cambios:
  - OpenAI: `POST /v1/chat/completions`
  - Anthropic: `POST /v1/messages` (y `/v1/messages/count_tokens`)
  - Gemini: `POST /v1beta/models/{modelo}:generateContent` y `:streamGenerateContent`
  - Imágenes (formato OpenAI): `POST /v1/images/generations`
- **Enrutado por reglas** (`rules.yaml`), en este orden:
  1. Contenido sensible (DNI, NIE, IBAN…) → siempre local. Si el local está caído, la petición falla en vez de salir.
  2. Etiquetas `@local` o `@frontera` en tu mensaje (se quitan antes de enviarlo).
  3. Una conversación que no cabe en el modelo local → nube (y las imágenes adjuntas, si `force_cloud_if.images` está a `true` porque tu modelo local no las entiende). PAIR lee de LM Studio el contexto real del modelo cargado.
  4. Palabras clave (resume, traduce… → local; diseña, depura… → nube).
  5. Modo: `ahorro` (local), `equilibrado` (local si tu último mensaje es corto; el historial no cuenta) o `calidad` (nube).
- **Generación de imágenes**: por defecto con tu generador local (`images.local_url`); con `@frontera`, en modo calidad o si el local falla, con OpenAI. Desde un chat, escribe `@imagen` y la descripción: la imagen llega como respuesta en la propia conversación, y en los mensajes siguientes PAIR adjunta la última imagen generada para que el modelo la vea ("descríbela", "¿qué le cambiarías?"). En los clientes con generador de imágenes aparece como el modelo `pair-image`.
- **Fallback**: si el backend elegido no responde, prueba con el otro (salvo contenido sensible).
- **Streaming y herramientas** (tool calls) en los tres formatos.
- **Registro** de cada decisión con su motivo, tokens, coste y ahorro.
- **Banco de pruebas**: guarda el último mensaje de cada petición no sensible para que puedas etiquetarlo después.

El ahorro se calcula con lo que habría costado la respuesta visible en la nube; el razonamiento interno de los modelos locales no cuenta.

Cada respuesta lleva las cabeceras `X-PAIR-Route` (destino y motivo) y `X-PAIR-Model`.

## Instalación

En macOS, un solo comando:

```bash
curl -fsSL https://raw.githubusercontent.com/ppernias/PAIR/main/install.sh | bash
```

El instalador descarga PAIR en `~/PAIR`, pregunta en qué equipo corre LM Studio (puede ser otro Mac de tu red, con *Serve on Local Network* activado), comprueba que responde, crea `config.yaml` y guarda tu clave de OpenAI en el llavero de macOS.

Instalación manual:

```bash
git clone https://github.com/ppernias/pair.git
cd pair
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml
export OPENAI_API_KEY=sk-...
```

En LM Studio, carga un modelo y arranca el servidor local (pestaña *Developer*, puerto 1234). Si dejas `local.model` vacío en `config.yaml`, PAIR usa el primer modelo cargado. Ajusta en `config.yaml` el modelo de OpenAI y su precio actual para que el cálculo de ahorro sea correcto.

## Uso

```bash
pair serve            # arranca el proxy en http://127.0.0.1:8787
pair log              # últimas decisiones
pair stats            # ahorro de los últimos 30 días
pair bench bench.csv  # exporta el banco de peticiones para etiquetarlo
```

Apunta tus clientes a PAIR cambiando su URL base:

| Cliente | Cómo |
| --- | --- |
| Cualquier cliente OpenAI (scripts, Open WebUI…) | URL base `http://127.0.0.1:8787/v1`, modelo `pair` |
| Claude Code | `export ANTHROPIC_BASE_URL=http://127.0.0.1:8787` |
| Clientes de Gemini que permitan cambiar la URL base | `http://127.0.0.1:8787` |

PAIR ignora la clave de API que envíe el cliente: usa siempre la suya, que busca en la variable `OPENAI_API_KEY` y, si no está, en el llavero de macOS (servicio `PAIR`).

## Fase 0: qué medir

1. Usa PAIR una o dos semanas en tu trabajo diario.
2. `pair bench bench.csv` y rellena la columna `etiqueta` con `local` o `cloud` (lo que *debería* haber pasado) en unas 200 peticiones.
3. Compara con la columna `decision_pair` y ajusta `rules.yaml`.
4. Puerta de paso a la fase 1: el ahorro estimado justifica seguir.

## Limitaciones conocidas de la fase 0

- No hay token de acceso: el proxy escucha solo en `127.0.0.1`.
- Los bloques de *thinking* de Anthropic y los documentos adjuntos se ignoran.
- Las respuestas de Gemini con herramientas se envían al final del stream, no en trozos.
- El *prompt caching* no se conserva: todo pasa por el formato de OpenAI.

## Desarrollo

```bash
pip install -e ".[dev]"
pytest
```

Los tests simulan LM Studio y OpenAI, así que no necesitan red ni claves.
