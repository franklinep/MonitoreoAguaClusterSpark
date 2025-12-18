import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.functions.vector_to_array

spark.sparkContext.setLogLevel("WARN")

val runTag = "cluster"  // "cluster" o "single"

case class ResultRow(
  modelo: String,
  run_tag: String,
  n_train: Long,
  n_test: Long,
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

  dfPred.withColumn("p_true", pTrue).withColumn("p_true_clipped", when(col("p_true") < lit(eps), lit(eps)).otherwise(col("p_true"))).withColumn("log_loss", -log(col("p_true_clipped"))).agg(avg(col("log_loss")).alias("log_loss")).first().getDouble(0)
}

def metricsFromPred(dfPred: org.apache.spark.sql.DataFrame): (Double, Double, Double) = {
  val predictionAndLabels = dfPred.select("prediction","label").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
  val mc = new MulticlassMetrics(predictionAndLabels)
  (mc.accuracy, mc.recall(1.0), mc.fMeasure(1.0))
}


val tAll0 = System.nanoTime()
val tPrep0 = System.nanoTime()

val indexer = new StringIndexer().setInputCol("tipo_punto").setOutputCol("tipo_punto_idx").setHandleInvalid("keep")

val encoder = new OneHotEncoder().setInputCol("tipo_punto_idx").setOutputCol("tipo_punto_oh").setHandleInvalid("keep")

val assembler = new VectorAssembler().setInputCols(Array("coord_norte","coord_este","anho","mes_num","tipo_punto_oh")).setOutputCol("features_raw").setHandleInvalid("skip")

val scaler = new StandardScaler().setInputCol("features_raw").setOutputCol("features").setWithStd(true).setWithMean(false)

val prepPipeline = new Pipeline().setStages(Array(indexer, encoder, assembler, scaler))
val prepModel = prepPipeline.fit(trainRaw)

val train = prepModel.transform(trainRaw).cache()
val test  = prepModel.transform(testRaw).cache()

val nTrain = train.count()
val nTest  = test.count()

val prepSec = (System.nanoTime() - tPrep0) / 1e9

val numFeatures = train.select("features").where(col("features").isNotNull).first().getAs[Vector]("features").size

println(s"\n[INFO][$runTag] nTrain=$nTrain nTest=$nTest numFeatures=$numFeatures")
println(f"[TIME][$runTag] prepSec=$prepSec%.3f\n")

// ======= MODELO MLCP =======
val tMLP0 = System.nanoTime()

val layers = Array(numFeatures, 16, 8, 2)

val tFitMLP0 = System.nanoTime()
val mlpc = new MultilayerPerceptronClassifier().setLabelCol("label").setFeaturesCol("features").setLayers(layers).setMaxIter(80).setBlockSize(128).setSeed(42L)

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
println(f"[MLPC][$runTag] accuracy=$accMLP%.4f recall_pos=$recallMLP%.4f f1_pos=$f1MLP%.4f logLoss=$lossMLP%.4f\n")

val rowMLP = ResultRow(
  "MLPC", runTag, nTrain, nTest,
  prepSec, fitMLPSec, predMLPSec, totalMLPSec,
  accMLP, recallMLP, f1MLP, lossMLP
)

// ======= RANDOM FOREST =======
val tRF0 = System.nanoTime()

val tFitRF0 = System.nanoTime()
val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(200).setMaxDepth(10).setSeed(42)

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
println(f"[RF][$runTag] accuracy=$accRF%.4f recall_pos=$recallRF%.4f f1_pos=$f1RF%.4f logLoss=$lossRF%.4f\n")

val rowRF = ResultRow(
  "RandomForest", runTag, nTrain, nTest,
  prepSec, fitRFSec, predRFSec, totalRFSec,
  accRF, recallRF, f1RF, lossRF
)

import spark.implicits._

val results = Seq(rowMLP, rowRF).toDF()

println("===================================================")
println(s" TABLA: Accuracy / Recall / F1 / Loss  [$runTag]")
println("===================================================")
results.select("modelo","run_tag","accuracy","recall_pos","f1_pos","log_loss").orderBy("modelo").show(false)

println("===================================================")
println(s" TABLA: Tiempos (sec) [$runTag]")
println("===================================================")
results.select("modelo","run_tag","prep_sec","fit_sec","pred_sec","total_sec").orderBy("modelo").show(false)

println(s"[DONE][$runTag]")
