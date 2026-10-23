#!/usr/bin/env bash
# Instalador de PAIR para macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/ppernias/PAIR/main/install.sh | bash
#
# Pregunta dónde está LM Studio y la clave de OpenAI (que guarda en el llavero),
# instala PAIR en ~/PAIR y deja config.yaml listo.
set -euo pipefail

DIR="${PAIR_DIR:-$HOME/PAIR}"
REPO="https://github.com/ppernias/PAIR.git"
say() { printf "\n\033[1m%s\033[0m\n" "$*"; }
ask() { local prompt="$1" default="$2" reply; read -r -p "$prompt [$default]: " reply </dev/tty; echo "${reply:-$default}"; }

say "1/5 Comprobando herramientas"
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Faltan las herramientas de línea de órdenes de Apple (git y python3)."
  echo "Se abrirá su instalador: acéptalo y, cuando termine, vuelve a ejecutar este comando."
  xcode-select --install || true
  exit 1
fi
PY=$(command -v python3)
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "PAIR necesita Python 3.11 o superior; tienes $("$PY" --version)."
  echo "Instálalo desde https://www.python.org/downloads/macos/ o con: brew install python@3.12"
  exit 1
}
echo "git y $("$PY" --version) disponibles"

say "2/5 Descargando PAIR en $DIR"
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull --ff-only; else git clone "$REPO" "$DIR"; fi
cd "$DIR"

say "3/5 Instalando dependencias"
"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e .
mkdir -p "$HOME/.local/bin"
ln -sf "$DIR/.venv/bin/pair" "$HOME/.local/bin/pair"

say "4/5 Conectando con LM Studio"
while true; do
  HOST=$(ask "Equipo donde corre LM Studio" "mac-studio.local")
  URL="http://$HOST:1234/v1"
  if MODELS=$(curl -fsS --max-time 5 "$URL/models" 2>/dev/null); then
    echo "LM Studio responde en $URL. Modelos: $(echo "$MODELS" | "$PY" -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))')"
    break
  fi
  echo "No hay respuesta en $URL. Comprueba que LM Studio tiene el servidor activo y 'Serve on Local Network'."
done

if [ ! -f config.yaml ]; then
  sed "s#http://localhost:1234/v1#$URL#" config.example.yaml > config.yaml
  echo "config.yaml creado"
else
  echo "config.yaml ya existía: no lo toco (base_url de LM Studio: $URL)"
fi

say "5/5 Clave de OpenAI"
if security find-generic-password -s PAIR -a OPENAI_API_KEY >/dev/null 2>&1; then
  echo "Ya hay una clave guardada en el llavero"
else
  read -r -s -p "Pega tu clave de OpenAI (no se mostrará): " KEY </dev/tty; echo
  security add-generic-password -U -s PAIR -a OPENAI_API_KEY -w "$KEY"
  echo "Clave guardada en el llavero de macOS (servicio PAIR)"
fi

say "PAIR instalado"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *)
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
  echo "Añadido ~/.local/bin al PATH en ~/.zshrc (abre una terminal nueva)";;
esac
echo "Arráncalo con:  cd $DIR && pair serve"
echo "Revisa antes en config.yaml el modelo de OpenAI y su precio."
