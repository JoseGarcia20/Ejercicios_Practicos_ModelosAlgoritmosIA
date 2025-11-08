from rdflib import RDF


def apply_context_axiom(g, ns):
    """
    Axioma: los dispositivos de la misma habitación comparten contexto.
    Materializamos añadiendo iot:comparteContextoCon entre dispositivos
    que comparten el mismo iot:ubicadoEn.
    """
    IOT = ns.IOT
    # Mapear habitacion -> lista de dispositivos
    room_to_devices = {}
    for dev, _, room in g.triples((None, IOT.ubicadoEn, None)):
        # Solo considerar entidades que sean Dispositivo
        if (dev, RDF.type, IOT.Dispositivo) in g:
            room_to_devices.setdefault(room, []).append(dev)

    # Por cada habitación, conectar pares con comparteContextoCon
    for room, devices in room_to_devices.items():
        n = len(devices)
        for i in range(n):
            for j in range(i + 1, n):
                a = devices[i]
                b = devices[j]
                g.add((a, IOT.comparteContextoCon, b))
                g.add((b, IOT.comparteContextoCon, a))

