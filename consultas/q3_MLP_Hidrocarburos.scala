/*
CC531 - Examen Final (Parte 2)
Consulta ML #1 (Multilayer Perceptron)

Objetivo (clasificación):
  Predecir si la concentración de "Hidrocarburos Totales de Petroleo (C10-C40)"
  en una muestra (punto + campaña + fecha/hora) es "ALTA" (>= percentil 75),
  usando otras mediciones fisicoquímicas (decimales) + coordenadas (decimales) y tiempo.

Ejecución (spark-shell):
  :load Q5_MLP_Hidrocarburos.scala

IMPORTANTE:
  - Edita las rutas HDFS (hdfsBase) según dónde pusiste tus JSON.
  - Cambia runTag ("cluster" vs "single") para no sobreescribir outputs.
*/

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Imputer, VectorAssembler, StandardScaler}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.functions.vector_to_array

// --------------------------------------------------------------------------------------
// Bloque para evitar colisiones de variables si cargas otros scripts en la misma sesión
// --------------------------------------------------------------------------------------
{
  spark.sparkContext.setLogLevel("WARN")

  // ============ (1) CONFIGURACIÓN DE RUTAS ============
  val hdfsBase = "hdfs:///data/monitoreo_agua"              // <-- EDITAR
  val pathFact   = s"$hdfsBase/fact_mediciones.json"        // newline-delimited JSON
  val pathCamp   = s"$hdfsBase/dim_campanas.json"
  val pathPuntos = s"$hdfsBase/dim_puntos.json"

  val runTag   = "cluster"                                  // <-- "cluster" o "single"
  val outMetrics = s"$hdfsBase/outputs/q5_mlp_metrics_$runTag"

  // Ajustes útiles (opcional)
  // spark.conf.set("spark.sql.shuffle.partitions", "200")
  // spark.conf.set("spark.sql.adaptive.enabled", "true")

  val tAll0 = System.nanoTime()

  // ============ (2) ESQUEMAS EXPLÍCITOS (evita inferSchema en datasets grandes) ============
  val factSchema = new StructType()
    .add("campaign_id", StringType, nullable = true)
    .add("point_id", IntegerType, nullable = true)
    .add("fecha_pto", StringType, nullable = true)
    .add("hora_pto", StringType, nullable = true)
    .add("parametro", StringType, nullable = true)
    .add("signo_parametro", StringType, nullable = true)
    .add("unidad_medida_parametro", StringType, nullable = true)
    .add("tipo_muestra", StringType, nullable = true)
    .add("resultado", DoubleType, nullable = true)
    .add("measurement_id", LongType, nullable = true)

  val puntosSchema = new StructType()
    .add("punto_muestreo", StringType, nullable = true)
    .add("tipo_punto", StringType, nullable = true)
    .add("txubigeo", StringType, nullable = true)
    .add("txzona", IntegerType, nullable = true)
    .add("coord_norte", DoubleType, nullable = true)
    .add("coord_este", DoubleType, nullable = true)
    .add("point_id", IntegerType, nullable = true)

  val campSchema = new StructType()
    .add("campaign_id", StringType, nullable = true)
    .add("cuc", StringType, nullable = true)
    .add("expediente", StringType, nullable = true)
    .add("txorigen", StringType, nullable = true)
    .add("coordinacion", StringType, nullable = true)
    .add("fechaini", StringType, nullable = true)
    .add("fechafin", StringType, nullable = true)
    .add("anho", IntegerType, nullable = true)
    .add("mes", StringType, nullable = true)
    .add("mes_num", IntegerType, nullable = true)
    .add("tarea_tdr", StringType, nullable = true)
    .add("matriz", StringType, nullable = true)
    .add("laboratorio_anonimizado", StringType, nullable = true)
    .add("rs", StringType, nullable = true)

  // ============ (3) LECTURA DESDE HDFS ============
  val tRead0 = System.nanoTime()
  val factRaw   = spark.read.schema(factSchema).json(pathFact)
  val puntosRaw = spark.read.schema(puntosSchema).json(pathPuntos)
  val campRaw   = spark.read.schema(campSchema).json(pathCamp)
  val readSec = (System.nanoTime() - tRead0) / 1e9
  println(f"[TIME] Lectura JSON: $readSec%.3f s")

  // ============ (4) LIMPIEZA / TRANSFORMACIÓN ============
  val params = Seq(
    "Hidrocarburos Totales de Petroleo (C10-C40)",
    "Aceites y Grasas",
    "Sólidos Totales Disueltos - STD",
    "Cloruros",
    "Hierro",
    "Aluminio",
    "Manganeso"
  )

  // Parseo de hora: "8H 45M 0S"
  val fact = factRaw
    .withColumn("parametro", trim(col("parametro")))
    .withColumn("fecha_pto_ts", to_timestamp(col("fecha_pto")))
    .withColumn("hora_h", regexp_extract(col("hora_pto"), "(\\d+)H", 1).cast("int"))
    .withColumn("hora_m", regexp_extract(col("hora_pto"), "(\\d+)M", 1).cast("int"))
    .withColumn("resultado_d", col("resultado").cast("double"))
    .filter(col("campaign_id").isNotNull && col("point_id").isNotNull && col("parametro").isNotNull && col("fecha_pto_ts").isNotNull)

  // Pivot (wide) por muestra (campaign_id, point_id, fecha/hora)
  val wide = fact
    .filter(col("parametro").isin(params: _*))
    .groupBy(col("campaign_id"), col("point_id"), col("fecha_pto_ts").alias("fecha_pto"), col("hora_h"), col("hora_m"))
    .pivot("parametro", params)
    .agg(first(col("resultado_d")))

  // Renombrar columnas a nombres "seguros"
  val wide2 = wide
    .withColumnRenamed("Hidrocarburos Totales de Petroleo (C10-C40)", "htp_c10_c40")
    .withColumnRenamed("Aceites y Grasas", "aceites_grasas")
    .withColumnRenamed("Sólidos Totales Disueltos - STD", "std")
    .withColumnRenamed("Cloruros", "cloruros")
    .withColumnRenamed("Hierro", "hierro")
    .withColumnRenamed("Aluminio", "aluminio")
    .withColumnRenamed("Manganeso", "manganeso")

  val puntos = puntosRaw.select(
    col("point_id"),
    col("coord_norte").cast("double"),
    col("coord_este").cast("double"),
    col("txzona").cast("int")
  )

  val camp = campRaw.select(
    col("campaign_id"),
    col("anho").cast("double"),
    col("mes_num").cast("double")
  )

  val df0 = wide2
    .join(puntos, Seq("point_id"), "left")
    .join(camp, Seq("campaign_id"), "left")
    .filter(col("htp_c10_c40").isNotNull) // label base no nula
    .cache()

  val n0 = df0.count() // materializa cache
  println(s"[INFO] Filas (muestras) para ML: $n0")

  // ============ (5) DEFINIR LABEL BINARIA (percentil 75 del target) ============
  val targetCol = "htp_c10_c40"
  val p75 = df0.stat.approxQuantile(targetCol, Array(0.75), 0.01)(0)
  println(f"[INFO] Umbral p75($targetCol) = $p75%.6f  -> label=1 si >= p75")

  val df = df0.withColumn("label", when(col(targetCol) >= lit(p75), 1.0).otherwise(0.0))

  // ============ (6) FEATURES (incluye decimales) ============
  val featureCols = Array(
    "aceites_grasas", "std", "cloruros", "hierro", "aluminio", "manganeso", // decimales (mg/L)
    "coord_norte", "coord_este",                                           // decimales (coordenadas UTM)
    "anho", "mes_num",                                                     // numéricos (tiempo)
    "hora_h", "hora_m"                                                     // numéricos (tiempo)
  )

  // Nos quedamos con columnas numéricas y label
  val dfML = df.select((featureCols.map(c => col(c).cast("double")) :+ col("label")): _*)
    .cache()
  dfML.count()

  val Array(train, test) = dfML.randomSplit(Array(0.8, 0.2), seed = 42L)
  val nTrain = train.count()
  val nTest  = test.count()
  println(s"[INFO] Split train/test: $nTrain / $nTest")

  // ============ (7) PIPELINE MLP ============
  val imputedCols = featureCols.map(_ + "_imp")

  val imputer = new Imputer()
    .setStrategy("median")
    .setInputCols(featureCols)
    .setOutputCols(imputedCols)

  val assembler = new VectorAssembler()
    .setInputCols(imputedCols)
    .setOutputCol("features")

  // Escalado recomendado para redes neuronales
  val scaler = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("features_scaled")
    .setWithMean(true)
    .setWithStd(true)

  val numFeatures = featureCols.length
  val layers = Array[Int](numFeatures, 16, 8, 2)

  val mlp = new MultilayerPerceptronClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features_scaled")
    .setLayers(layers)
    .setMaxIter(120)
    .setSeed(42L)
    .setBlockSize(128)

  val pipeline = new Pipeline().setStages(Array(imputer, assembler, scaler, mlp))

  // ============ (8) ENTRENAR + EVALUAR ============
  val tFit0 = System.nanoTime()
  val model = pipeline.fit(train)
  val fitSec = (System.nanoTime() - tFit0) / 1e9
  println(f"[TIME] MLP fit(): $fitSec%.3f s")

  val tPred0 = System.nanoTime()
  val pred = model.transform(test).cache()
  val predN = pred.count()
  val predSec = (System.nanoTime() - tPred0) / 1e9
  println(f"[TIME] MLP transform()+count(): $predSec%.3f s (n=$predN)")

  // Métricas: accuracy / precision / recall / f1 (clase positiva=1)
  val predictionAndLabels = pred.select("prediction", "label").rdd.map { r =>
    (r.getDouble(0), r.getDouble(1))
  }
  val mc = new MulticlassMetrics(predictionAndLabels)

  val accuracy    = mc.accuracy
  val precision1  = mc.precision(1.0)
  val recall1     = mc.recall(1.0)
  val f1_1        = mc.fMeasure(1.0)

  // Pérdida (log-loss) promedio en test: -log(p(y_true))
  val eps = 1e-15
  val probArr = vector_to_array(col("probability"))
  val pTrue = when(col("label") === 1.0, probArr.getItem(1)).otherwise(probArr.getItem(0))

  val logLoss = pred
    .withColumn("p_true", pTrue)
    .withColumn("p_true_clipped", when(col("p_true") < lit(eps), lit(eps)).otherwise(col("p_true")))
    .withColumn("log_loss", -log(col("p_true_clipped")))
    .agg(avg(col("log_loss")).alias("log_loss"))
    .first()
    .getDouble(0)

  println(f"[METRICS][MLP] accuracy=$accuracy%.4f  precision_pos=$precision1%.4f  recall_pos=$recall1%.4f  f1_pos=$f1_1%.4f  logLoss=$logLoss%.4f")

  // ============ (9) GUARDAR MÉTRICAS EN HDFS (para luego armar tabla comparativa) ============
  import spark.implicits._
  val totalSec = (System.nanoTime() - tAll0) / 1e9

  val metricsDF = Seq(
    (
      "MLP",
      runTag,
      spark.sparkContext.master,
      spark.sparkContext.applicationId,
      n0,
      nTrain,
      nTest,
      p75,
      readSec,
      fitSec,
      predSec,
      totalSec,
      accuracy,
      precision1,
      recall1,
      f1_1,
      logLoss
    )
  ).toDF(
    "modelo","run_tag","master","app_id","n_samples","n_train","n_test","threshold_p75",
    "read_sec","fit_sec","pred_sec","total_sec",
    "accuracy","precision_pos","recall_pos","f1_pos","log_loss"
  )

  metricsDF.show(truncate = false)
  metricsDF.coalesce(1).write.mode("overwrite").json(outMetrics)

  println(s"[INFO] Métricas guardadas en: $outMetrics")
  println(f"[TIME] TOTAL script: $totalSec%.3f s")
}
