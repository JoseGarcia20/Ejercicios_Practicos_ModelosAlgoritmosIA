import argparse
from rdflib import Graph

from src.ontologia_iot.ontology import build_graph, simulate
from src.ontologia_iot.reasoner import apply_context_axiom
from src.ontologia_iot.queries import EXAMPLE_QUERIES


def _pretty_node(g: Graph, node):
    # 1) iot:nombre si existe
    try:
        IOT = g.namespace_manager.compute_qname("http://example.org/iot#")[0]  # ensure ns registered
    except Exception:
        IOT = None
    try:
        nombre_pred = g.namespace_manager.expand_curie("iot:nombre")
    except Exception:
        nombre_pred = None
    if nombre_pred is not None:
        nombre = next(g.objects(node, nombre_pred), None)
        if nombre is not None:
            return str(nombre)
    # 2) qname con prefijo (iot:Recurso)
    try:
        return g.namespace_manager.qname(node)
    except Exception:
        s = str(node)
        # 3) fragmento local tras '#' o última '/'
        if "#" in s:
            return s.rsplit("#", 1)[-1]
        return s.rsplit("/", 1)[-1]


def run_demo():
    g, ns = build_graph()
    apply_context_axiom(g, ns)

    print("[Demo] Ontología construida y axiomas aplicados.")
    simulate(g, ns)

    print("\n[Demo] Consultas de ejemplo:")
    for title, query in EXAMPLE_QUERIES.items():
        print(f"\n# {title}")
        for row in g.query(query):
            print(tuple(_pretty_node(g, x) for x in row))


def run_repl():
    g, ns = build_graph()
    apply_context_axiom(g, ns)

    print("SPARQL REPL. Escriba 'help' para ver ejemplos, 'exit' para salir.")
    print("Consejo: los resultados muestran nombres legibles cuando hay iot:nombre.")
    while True:
        try:
            line = input("sparql> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break
        if line.lower() == "help":
            print("Ejemplos disponibles:")
            for title, query in EXAMPLE_QUERIES.items():
                print(f"\n# {title}\n{query}")
            continue
        # Read multiline queries ending with ;;;
        if not line.endswith(";;;"):
            lines = [line]
            while True:
                more = input("...     ")
                lines.append(more)
                if more.strip().endswith(";;;"):
                    break
            line = "\n".join(lines)
        q = line[:-3]  # strip trailing ';;;' terminator
        try:
            results = g.query(q)
            for row in results:
                print(tuple(_pretty_node(g, x) for x in row))
        except Exception as e:
            print(f"Error ejecutando consulta: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ontología IoT para hogar inteligente (SPARQL)")
    parser.add_argument("--demo", action="store_true", help="Ejecuta demo con consultas de ejemplo")
    parser.add_argument("--repl", action="store_true", help="Abre consola SPARQL interactiva")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.repl:
        run_repl()
    else:
        print("Use --demo para una demostración o --repl para la consola SPARQL.")


if __name__ == "__main__":
    main()
