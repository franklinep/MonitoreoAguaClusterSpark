import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.functions.vector_to_array

{
  spark.sparkContext.setLogLevel("WARN")

  val runTag = "cluster"  // <-- CAMBIA A "single" cuando corras en 1 nodo

  val cargarDesdeHDFS = false


  // Parámetros de modelos
  val mlpMaxIter = 80
  val rfNumTrees = 200
  val rfMaxDepth = 10


  def line(ch: String = "=", n: Int = 80): Unit = { print(ch * n); print("\n") }
  def section(title: String): Unit = { line("="); println(title); line("=") }
  def sub(title: String): Unit = { line("-"); println(title); line("-") }

  case class ResultRow(
    modelo: String,
    run_tag: String,
    n_train: Long,
    n_test: Long,
    q75: Double,
    prep_sec: Double,
    fit_sec: Double,
    pred_sec: Double,
    total_sec: Double,
    accuracy: Double,
    recall_pos: Double,
    f1_pos: Double,
    log_loss: Double
  )

  def logLossFromPred(dfPred: org.apache.spark.sql.DataFrame): Double = {
    val eps = 1e-15
    val probArr = vector_to_array(col("probability"))
    val pTrue = when(col("label") === 1.0, probArr.getItem(1)).otherwise(probArr.getItem(0))

    dfPred
      .withColumn("p_true", pTrue)
      .withColumn("p_true_clipped", when(col("p_true") < lit(eps), lit(eps)).otherwise(col("p_true")))
      .withColumn("log_loss", -log(col("p_true_clipped")))
      .agg(avg(col("log_loss")).alias("log_loss"))
      .first().getDouble(0)
  }

  def metricsFromPred(dfPred: org.apache.spark.sql.DataFrame): (Double, Double, Double) = {
    val predictionAndLabels = dfPred.select("prediction","label").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    val mc = new MulticlassMetrics(predictionAndLabels)
    (mc.accuracy, mc.recall(1.0), mc.fMeasure(1.0))
  }

  section(s"[1] PREPARACIÓN DE DATOS [$runTag]")
  @transient val medicionesDF = mediciones
  @transient val puntosDF = puntos
  @transient val campanasDF = campanas

  val baseML = medicionesDF
    .filter(col("tipo_muestra") === "metales_totales")
    .withColumn("resultado", col("resultado").cast("double"))
    .filter(col("resultado").isNotNull && !isnan(col("resultado")))
    .join(puntosDF.select(
      col("point_id"),
      col("coord_norte").cast("double").alias("coord_norte"),
      col("coord_este").cast("double").alias("coord_este"),
      col("tipo_punto")
    ), Seq("point_id"), "inner")
    .join(campanasDF.select(
      col("campaign_id"),
      col("anho").cast("double").alias("anho"),
      col("mes_num").cast("double").alias("mes_num")
    ), Seq("campaign_id"), "inner")
    .filter(col("coord_norte").isNotNull && !isnan(col("coord_norte")))
    .filter(col("coord_este").isNotNull && !isnan(col("coord_este")))
    .withColumn("tipo_punto",
      when(col("tipo_punto").isNull || length(trim(col("tipo_punto"))) === 0, lit("DESCONOCIDO"))
        .otherwise(trim(col("tipo_punto")))
    )
    .select("resultado","coord_norte","coord_este","anho","mes_num","tipo_punto")
    .cache()

  val nBase = baseML.count()
  println(s"[INFO][$runTag] baseML rows = $nBase")

  val Array(trainBaseML, testBaseML) = baseML.randomSplit(Array(0.8, 0.2), seed = 42L)
  trainBaseML.cache(); testBaseML.cache()

  val q75 = trainBaseML.stat.approxQuantile("resultado", Array(0.75), 0.01)(0)

  val trainRaw = trainBaseML.withColumn("label", when(col("resultado") >= lit(q75), 1.0).otherwise(0.0)).cache()
  val testRaw  = testBaseML.withColumn("label",  when(col("resultado") >= lit(q75), 1.0).otherwise(0.0)).cache()

  val nTrainRaw = trainRaw.count()
  val nTestRaw  = testRaw.count()

  println(f"[INFO][$runTag] q75(train) = $q75%.8f")
  println(s"[INFO][$runTag] trainRaw = $nTrainRaw, testRaw = $nTestRaw")

  sub(s"[CHECK][$runTag] Distribución de clases (label)")
  trainRaw.groupBy("label").count().orderBy("label").show(false)
  testRaw.groupBy("label").count().orderBy("label").show(false)

  // =====================
  // FEATURE ENGINEERING
  // =====================
  section(s"[2] FEATURE ENGINEERING [$runTag]")

  val tPrep0 = System.nanoTime()

  val indexer = new StringIndexer()
    .setInputCol("tipo_punto")
    .setOutputCol("tipo_punto_idx")
    .setHandleInvalid("keep")

  val encoder = new OneHotEncoder()
    .setInputCol("tipo_punto_idx")
    .setOutputCol("tipo_punto_oh")
    .setHandleInvalid("keep")

  val assembler = new VectorAssembler()
    .setInputCols(Array("coord_norte","coord_este","anho","mes_num","tipo_punto_oh"))
    .setOutputCol("features_raw")
    .setHandleInvalid("skip")

  val scaler = new StandardScaler()
    .setInputCol("features_raw")
    .setOutputCol("features")
    .setWithStd(true)
    .setWithMean(false)

  val prepPipeline = new Pipeline().setStages(Array(indexer, encoder, assembler, scaler))
  val prepModel = prepPipeline.fit(trainRaw)

  val train = prepModel.transform(trainRaw).cache()
  val test  = prepModel.transform(testRaw).cache()

  val nTrain = train.count()
  val nTest  = test.count()

  val prepSec = (System.nanoTime() - tPrep0) / 1e9
  val numFeatures = train.select("features").where(col("features").isNotNull).first().getAs[Vector]("features").size

  println(s"[INFO][$runTag] nTrain=$nTrain nTest=$nTest numFeatures=$numFeatures")
  println(f"[TIME][$runTag] prepSec=$prepSec%.3f")

  // =====================
  // MODELO 1: MLPC
  // =====================
  section(s"[3] MLPC [$runTag]")

  val tMLP0 = System.nanoTime()

  val layers = Array(numFeatures, 16, 8, 2)

  val tFitMLP0 = System.nanoTime()
  val mlpc = new MultilayerPerceptronClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setLayers(layers)
    .setMaxIter(mlpMaxIter)
    .setBlockSize(128)
    .setSeed(42L)

  val mlpcModel = mlpc.fit(train)
  val fitMLPSec = (System.nanoTime() - tFitMLP0) / 1e9

  val tPredMLP0 = System.nanoTime()
  val predMLP = mlpcModel.transform(test).cache()
  val nPredMLP = predMLP.count()
  val predMLPSec = (System.nanoTime() - tPredMLP0) / 1e9

  val (accMLP, recallMLP, f1MLP) = metricsFromPred(predMLP)
  val lossMLP = logLossFromPred(predMLP)

  val totalMLPSec = (System.nanoTime() - tMLP0) / 1e9

  println(f"[MLPC][$runTag] nPred=$nPredMLP fitSec=$fitMLPSec%.3f predSec=$predMLPSec%.3f totalSec=$totalMLPSec%.3f")
  println(f"[MLPC][$runTag] accuracy=$accMLP%.4f recall_pos=$recallMLP%.4f f1_pos=$f1MLP%.4f logLoss=$lossMLP%.4f")

  val rowMLP = ResultRow(
    "MLPC", runTag, nTrain, nTest, q75,
    prepSec, fitMLPSec, predMLPSec, totalMLPSec,
    accMLP, recallMLP, f1MLP, lossMLP
  )

  // =====================
  // MODELO 2: RandomForest
  // =====================
  section(s"[4] RandomForest [$runTag]")

  val tRF0 = System.nanoTime()

  val tFitRF0 = System.nanoTime()
  val rf = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setNumTrees(rfNumTrees)
    .setMaxDepth(rfMaxDepth)
    .setSeed(42)

  val rfModel = rf.fit(train)
  val fitRFSec = (System.nanoTime() - tFitRF0) / 1e9

  val tPredRF0 = System.nanoTime()
  val predRF = rfModel.transform(test).cache()
  val nPredRF = predRF.count()
  val predRFSec = (System.nanoTime() - tPredRF0) / 1e9

  val (accRF, recallRF, f1RF) = metricsFromPred(predRF)
  val lossRF = logLossFromPred(predRF)

  val totalRFSec = (System.nanoTime() - tRF0) / 1e9

  println(f"[RF][$runTag] nPred=$nPredRF fitSec=$fitRFSec%.3f predSec=$predRFSec%.3f totalSec=$totalRFSec%.3f")
  println(f"[RF][$runTag] accuracy=$accRF%.4f recall_pos=$recallRF%.4f f1_pos=$f1RF%.4f logLoss=$lossRF%.4f")

  val rowRF = ResultRow(
    "RandomForest", runTag, nTrain, nTest, q75,
    prepSec, fitRFSec, predRFSec, totalRFSec,
    accRF, recallRF, f1RF, lossRF
  )

  // =====================
  // TABLAS
  // =====================
  section(s"[5] TABLAS [$runTag]")

  import spark.implicits._
  val results = Seq(rowMLP, rowRF).toDF()

  sub(s"TABLA 2: Accuracy / Recall / F1 / Loss  [$runTag]")
  results
    .select("modelo","run_tag","accuracy","recall_pos","f1_pos","log_loss")
    .orderBy("modelo")
    .show(false)

  sub(s"TABLA 1: Tiempos (sec)  [$runTag]")
  results
    .select("modelo","run_tag","prep_sec","fit_sec","pred_sec","total_sec")
    .orderBy("modelo")
    .show(false)

  line("=")
  println(s"[DONE][$runTag].")
  line("=")
}
