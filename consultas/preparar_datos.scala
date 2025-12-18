import org.apache.spark.sql.functions._

// =====================
// 1) Base ML (joins + filtros)
// =====================
val baseML = mediciones
  .filter(col("resultado").isNotNull && !isnan(col("resultado")))
  .filter(col("tipo_muestra") === "metales_totales")
  .join(puntos.select("point_id","coord_norte","coord_este","tipo_punto"), Seq("point_id"), "inner")
  .join(campanas.select("campaign_id","anho","mes_num"), Seq("campaign_id"), "inner")
  .select(
  col("resultado").cast("double"),
  col("coord_norte").cast("double"),
  col("coord_este").cast("double"),
  col("anho").cast("double"),
  col("mes_num").cast("double"),
  col("tipo_punto")
)

val Array(trainBaseML, testBaseML) = baseML.randomSplit(Array(0.8, 0.2), seed = 42)
trainBaseML.cache()
testBaseML.cache()

// =====================
// 2) Label: umbral q75 SOLO en train (evita leakage)
// =====================
val q75 = trainBaseML.stat.approxQuantile("resultado", Array(0.75), 0.01)(0)

val trainRaw = trainBaseML.withColumn("label", when(col("resultado") >= lit(q75), 1.0).otherwise(0.0))
val testRaw  = testBaseML.withColumn("label",  when(col("resultado") >= lit(q75), 1.0).otherwise(0.0))

println(s"[MLPC] q75(train) = $q75")
println(s"[MLPC] train = ${trainRaw.count()}, test = ${testRaw.count()}")
trainRaw.printSchema()
