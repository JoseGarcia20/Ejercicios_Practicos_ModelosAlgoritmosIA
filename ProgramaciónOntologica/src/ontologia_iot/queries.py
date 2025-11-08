EXAMPLE_QUERIES = {
    "Dispositivos por habitación": (
        """
PREFIX iot: <http://example.org/iot#>
SELECT ?hab ?dis WHERE {
  ?dis a iot:Dispositivo ;
       iot:ubicadoEn ?hab .
}
        """.strip()
    )
}
