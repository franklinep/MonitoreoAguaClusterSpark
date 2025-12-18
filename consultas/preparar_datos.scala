import org.apache.spark.sql.functions._

spark.sparkContext.setLogLevel("WARN")


val puntosSel = puntos
  .select(
    col("point_id"),
    col("coord_norte").cast("double").alias("coord_norte"),
    col("coord_este").cast("double").alias("coord_este"),
    col("tipo_punto").alias("tipo_punto")
  )

val campanasSel = campanas
  .select(
    col("campaign_id"),
    col("anho").cast("double").alias("anho"),
    col("mes_num").cast("double").alias("mes_num")
  )

val baseML = mediciones
  .filter(col("tipo_muestra") === "metales_totales")
  .withColumn("resultado", col("resultado").cast("double"))
  .filter(col("resultado").isNotNull && !isnan(col("resultado")))
  .join(puntosSel, Seq("point_id"), "inner")
  .join(campanasSel, Seq("campaign_id"), "inner")
  .filter(col("coord_norte").isNotNull && !isnan(col("coord_norte")))
  .filter(col("coord_este").isNotNull && !isnan(col("coord_este")))
  .withColumn(
    "tipo_punto",
    when(col("tipo_punto").isNull || length(trim(col("tipo_punto"))) === 0, lit("DESCONOCIDO"))
      .otherwise(trim(col("tipo_punto")))
  )
  .select(
    col("resultado"),
    col("coord_norte"),
    col("coord_este"),
    col("anho"),
    col("mes_num"),
    col("tipo_punto")
  )
  .cache()

println(s"[PREP] baseML rows = ${baseML.count()}")

baseML.groupBy("tipo_punto").count().orderBy(desc("count")).show(20, false)

// =====================================================
// 2) Split train/test
// =====================================================
val Array(trainBaseML, testBaseML) = baseML.randomSplit(Array(0.8, 0.2), seed = 42L)
trainBaseML.cache()
testBaseML.cache()

println(s"[PREP] trainBaseML = ${trainBaseML.count()}, testBaseML = ${testBaseML.count()}")

// =====================================================
// 3) Label con q75 SOLO en train (sin leakage)
// =====================================================
val q75 = trainBaseML.stat.approxQuantile("resultado", Array(0.75), 0.01)(0)

val trainRaw = trainBaseML.withColumn("label", when(col("resultado") >= lit(q75), 1.0).otherwise(0.0)).cache()
val testRaw  = testBaseML.withColumn("label",  when(col("resultado") >= lit(q75), 1.0).otherwise(0.0)).cache()

println(s"[PREP] q75(train) = $q75")
println(s"[PREP] trainRaw = ${trainRaw.count()}, testRaw = ${testRaw.count()}")

trainRaw.printSchema()
testRaw.printSchema()

trainRaw.groupBy("label").count().orderBy("label").show(false)
testRaw.groupBy("label").count().orderBy("label").show(false)
