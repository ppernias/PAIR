"""Línea de órdenes: pair serve | stats | log | bench."""

from __future__ import annotations

import argparse
import sys

from . import config as config_mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pair", description="PAIR — Personal AI Router")
    parser.add_argument("-c", "--config", default="config.yaml", help="ruta a config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="arranca el proxy")
    p_serve.add_argument("--debug", action="store_true",
                         help="muestra la estructura de cada petición y guarda la última en ~/.pair/last_request.json")
    p_stats = sub.add_parser("stats", help="resumen de ahorro")
    p_stats.add_argument("--days", type=float, default=30)
    p_log = sub.add_parser("log", help="últimas decisiones")
    p_log.add_argument("-n", type=int, default=20)
    p_bench = sub.add_parser("bench", help="exporta el banco de peticiones a CSV para etiquetarlo")
    p_bench.add_argument("out", nargs="?", default="bench.csv")

    args = parser.parse_args(argv)
    cfg = config_mod.load(args.config)

    if args.command == "serve":
        import uvicorn

        from .app import create_app

        if not cfg.cloud.api_key:
            print(f"Aviso: no hay clave en la variable {cfg.cloud.api_key_env} ni en el llavero (servicio PAIR);"
                  " las peticiones a la nube fallarán.", file=sys.stderr)
        print(f"PAIR escuchando en http://{cfg.host}:{cfg.port}  (modo {cfg.mode})")
        uvicorn.run(create_app(cfg, debug=args.debug, echo=True), host=cfg.host, port=cfg.port, log_level="warning")
        return 0

    from .store import Store

    store = Store(cfg.log_path, cfg.cloud.price_input_per_mtok, cfg.cloud.price_output_per_mtok)
    if args.command == "stats":
        s = store.stats(args.days)
        print(f"Últimos {s['days']:g} días: {s['requests']} peticiones")
        print(f"  Resueltas en local: {s['local']} ({s['local_pct']}%)")
        print(f"  Con fallback: {s['fell_back']} · sensibles: {s['sensitive']} · errores: {s['errors']}")
        print(f"  Coste real: {s['cost_usd']:.4f} USD · todo a la nube: {s['all_cloud_usd']:.4f} USD")
        print(f"  Ahorro: {s['saved_usd']:.4f} USD ({s['saved_pct']}%)")
    elif args.command == "log":
        from .consola import linea

        for row in store.recent(args.n):
            print(linea(*row))
    elif args.command == "bench":
        n = store.export_bench(args.out)
        print(f"{n} peticiones exportadas a {args.out}. Rellena la columna 'etiqueta' con local o cloud.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
