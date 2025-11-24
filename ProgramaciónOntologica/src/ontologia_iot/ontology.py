from rdflib import Graph, Namespace, Literal, RDF, RDFS, OWL, XSD, URIRef


def build_graph():
    g = Graph()
    IOT = Namespace("http://example.org/iot#")
    g.bind("iot", IOT)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    # Clases
    classes = {
        "Dispositivo": IOT.Dispositivo,
        "Sensor": IOT.Sensor,
        "Actuador": IOT.Actuador,
        "Habitacion": IOT.Habitacion,
        "Automatizacion": IOT.Automatizacion,
        "Usuario": IOT.Usuario,
        "Evento": IOT.Evento,
        "Movimiento": IOT.Movimiento,
        "Temperatura": IOT.Temperatura,
    }
    for c in classes.values():
        g.add((c, RDF.type, OWL.Class))
    # Jerarquía
    g.add((IOT.Sensor, RDFS.subClassOf, IOT.Dispositivo))
    g.add((IOT.Actuador, RDFS.subClassOf, IOT.Dispositivo))
    g.add((IOT.Movimiento, RDFS.subClassOf, IOT.Evento))
    g.add((IOT.Temperatura, RDFS.subClassOf, IOT.Evento))

    # Propiedades de objeto
    def op(prop, domain=None, range_=None, types=(OWL.ObjectProperty,)):
        for t in types:
            g.add((prop, RDF.type, t))
        if domain is not None:
            g.add((prop, RDFS.domain, domain))
        if range_ is not None:
            g.add((prop, RDFS.range, range_))

    op(IOT.ubicadoEn, IOT.Dispositivo, IOT.Habitacion)
    op(IOT.controla, IOT.Automatizacion, IOT.Dispositivo)
    op(IOT.detecta, IOT.Sensor, IOT.Evento)
    op(IOT.desencadena, IOT.Evento, IOT.Automatizacion)
    op(IOT.perteneceA, IOT.Habitacion, IOT.Usuario)
    # comparteContextoCon es simétrica
    op(IOT.comparteContextoCon, IOT.Dispositivo, IOT.Dispositivo, (OWL.ObjectProperty, OWL.SymmetricProperty))

    # Propiedades de datos
    def dp(prop, domain=None, range_=None):
        g.add((prop, RDF.type, OWL.DatatypeProperty))
        if domain is not None:
            g.add((prop, RDFS.domain, domain))
        if range_ is not None:
            g.add((prop, RDFS.range, range_))

    dp(IOT.tieneEstado, IOT.Dispositivo, XSD.string)  # e.g., "On"/"Off"
    dp(IOT.valor, IOT.Sensor, XSD.decimal)            # lectura numérica
    dp(IOT.nombre, None, XSD.string)

    # Individuos (habitaciones)
    sala = IOT.Sala
    dormitorio = IOT.Dormitorio
    g.add((sala, RDF.type, IOT.Habitacion))
    g.add((sala, IOT.nombre, Literal("Sala", datatype=XSD.string)))
    g.add((dormitorio, RDF.type, IOT.Habitacion))
    g.add((dormitorio, IOT.nombre, Literal("Dormitorio", datatype=XSD.string)))

    # Usuarios
    alice = IOT.Alice
    bob = IOT.Bob
    g.add((alice, RDF.type, IOT.Usuario))
    g.add((bob, RDF.type, IOT.Usuario))
    g.add((sala, IOT.perteneceA, alice))
    g.add((dormitorio, IOT.perteneceA, bob))

    # Dispositivos: sensores y actuadores
    luz_sala = IOT.LuzSala
    sensor_mov_sala = IOT.SensorMovimientoSala
    luz_dorm = IOT.LuzDormitorio
    termostato_dorm = IOT.TermostatoDormitorio
    calefactor_dorm = IOT.CalefactorDormitorio
    aire_ac_dorm = IOT.AireAcondicionadoDormitorio

    for dev in [luz_sala, sensor_mov_sala, luz_dorm, termostato_dorm, calefactor_dorm, aire_ac_dorm]:
        g.add((dev, RDF.type, IOT.Dispositivo))
        g.add((dev, IOT.tieneEstado, Literal("Off", datatype=XSD.string)))

    g.add((sensor_mov_sala, RDF.type, IOT.Sensor))
    g.add((termostato_dorm, RDF.type, IOT.Sensor))
    g.add((luz_sala, RDF.type, IOT.Actuador))
    g.add((luz_dorm, RDF.type, IOT.Actuador))
    g.add((calefactor_dorm, RDF.type, IOT.Actuador))
    g.add((aire_ac_dorm, RDF.type, IOT.Actuador))

    g.add((luz_sala, IOT.ubicadoEn, sala))
    g.add((sensor_mov_sala, IOT.ubicadoEn, sala))
    g.add((luz_dorm, IOT.ubicadoEn, dormitorio))
    g.add((termostato_dorm, IOT.ubicadoEn, dormitorio))
    g.add((calefactor_dorm, IOT.ubicadoEn, dormitorio))
    g.add((aire_ac_dorm, IOT.ubicadoEn, dormitorio))

    # Nombres legibles de dispositivos
    g.add((luz_sala, IOT.nombre, Literal("Luz Sala", datatype=XSD.string)))
    g.add((sensor_mov_sala, IOT.nombre, Literal("Sensor Movimiento Sala", datatype=XSD.string)))
    g.add((luz_dorm, IOT.nombre, Literal("Luz Dormitorio", datatype=XSD.string)))
    g.add((termostato_dorm, IOT.nombre, Literal("Termostato Dormitorio", datatype=XSD.string)))
    g.add((calefactor_dorm, IOT.nombre, Literal("Calefactor Dormitorio", datatype=XSD.string)))
    g.add((aire_ac_dorm, IOT.nombre, Literal("Aire Acondicionado Dormitorio", datatype=XSD.string)))

    # Sensores detectan tipos de evento
    g.add((sensor_mov_sala, IOT.detecta, IOT.Movimiento))
    g.add((termostato_dorm, IOT.detecta, IOT.Temperatura))

    # Automatizaciones
    auto_luz_sala = IOT.AutoLuzSala
    auto_clima_dorm = IOT.AutoClimaDorm
    auto_fresco_dorm = IOT.AutoFrescoDorm
    for a in [auto_luz_sala, auto_clima_dorm, auto_fresco_dorm]:
        g.add((a, RDF.type, IOT.Automatizacion))
    g.add((auto_luz_sala, IOT.nombre, Literal("Auto Luz Sala", datatype=XSD.string)))
    g.add((auto_clima_dorm, IOT.nombre, Literal("Auto Clima Dormitorio", datatype=XSD.string)))
    g.add((auto_fresco_dorm, IOT.nombre, Literal("Auto Fresco Dormitorio", datatype=XSD.string)))

    # Auto luz sala: si hay movimiento en sala, encender luz sala
    g.add((auto_luz_sala, IOT.controla, luz_sala))
    # Auto clima dormitorio (frío): mantener 22C -> controla calefactor
    g.add((auto_clima_dorm, IOT.controla, calefactor_dorm))
    # Auto fresco dormitorio (calor): si hace mucho calor -> controla aire acondicionado
    g.add((auto_fresco_dorm, IOT.controla, aire_ac_dorm))

    # Evento real de movimiento en sala que dispara AutoLuzSala
    ev_mov_sala = IOT.EventoMovimientoSala1
    g.add((ev_mov_sala, RDF.type, IOT.Movimiento))
    g.add((ev_mov_sala, IOT.ubicadoEn, sala))
    g.add((ev_mov_sala, IOT.desencadena, auto_luz_sala))

    # Valor de temperatura en dormitorio (subido para disparar caso "calor")
    g.add((termostato_dorm, IOT.valor, Literal(29.5)))

    ns = type("NS", (), {"IOT": IOT})
    return g, ns


def simulate(g: Graph, ns):
    IOT = ns.IOT
    # Simula: procesa eventos que "desencadenan" automatizaciones
    for ev, _, auto in g.triples((None, IOT.desencadena, None)):
        # Para cada automatización, enciende los dispositivos que controla
        for _, _, disp in g.triples((auto, IOT.controla, None)):
            g.set((disp, IOT.tieneEstado, Literal("On", datatype=XSD.string)))

    # Lógica muy sencilla de clima: si temperatura < 22 -> enciende calefactor
    target_temp = 22.0
    for sensor in g.subjects(RDF.type, IOT.Sensor):
        for _, _, val in g.triples((sensor, IOT.valor, None)):
            try:
                temp = float(val)
            except Exception:
                continue
            if temp < target_temp:
                # encuentra automatizaciones que controlen calefactor
                for auto in g.subjects(RDF.type, IOT.Automatizacion):
                    for _, _, dev in g.triples((auto, IOT.controla, None)):
                        # Heurística: activar si el dispositivo está en la misma habitación
                        room = next(g.objects(dev, IOT.ubicadoEn), None)
                        sroom = next(g.objects(sensor, IOT.ubicadoEn), None)
                        if room and sroom and room == sroom:
                            g.set((dev, IOT.tieneEstado, Literal("On", datatype=XSD.string)))
