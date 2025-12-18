puntos.createOrReplaceTempView("puntos")
campanas.createOrReplaceTempView("campanas")
mediciones.createOrReplaceTempView("mediciones")

val q2 = spark.sql("""
SELECT
  c.anho,
  c.mes_num,
  p.punto_muestreo,
  p.txzona,
  p.coord_norte,
  p.coord_este,
  m.parametro,
  m.tipo_muestra,
  m.resultado
FROM mediciones m
JOIN campanas c ON m.campaign_id = c.campaign_id
JOIN puntos p   ON m.point_id = p.point_id
WHERE m.resultado IS NOT NULL
  AND m.tipo_muestra = 'metales_totales'
  AND c.anho >= 2020
ORDER BY m.resultado DESC
LIMIT 50
""")

q2.show(50, truncate=false)
