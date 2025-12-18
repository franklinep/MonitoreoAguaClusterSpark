import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler}
import org.apache.spark.ml.linalg.Vector

// =====================
// 3) Feature engineering
// =====================
val indexer = new StringIndexer()
  .setInputCol("tipo_punto")
  .setOutputCol("tipo_punto_idx")
  .setHandleInvalid("keep")

val encoder = new OneHotEncoder()
  .setInputCol("tipo_punto_idx")
  .setOutputCol("tipo_punto_oh")

val assembler = new VectorAssembler()
  .setInputCols(Array("coord_norte","coord_este","anho","mes_num","tipo_punto_oh"))
  .setOutputCol("features_raw")

val scaler = new StandardScaler()
  .setInputCol("features_raw")
  .setOutputCol("features")
  .setWithStd(true)
  .setWithMean(false)

// Fit SOLO con train
val prepPipeline = new Pipeline().setStages(Array(indexer, encoder, assembler, scaler))
val prepModel = prepPipeline.fit(trainRaw)

val train = prepModel.transform(trainRaw).cache()
val test  = prepModel.transform(testRaw).cache()

val numFeatures = train
  .select("features")
  .where(col("features").isNotNull)
  .first()
  .getAs[Vector]("features")
  .size

val layers = Array(numFeatures, 16, 8, 2)

// =====================
// 4) Modelo MLPC
// =====================
val mlpc = new MultilayerPerceptronClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setLayers(layers)
  .setMaxIter(80)
  .setBlockSize(128)
  .setSeed(42L)

val mlpcModel = mlpc.fit(train)
val pred = mlpcModel.transform(test)

// Accuracy (luego tú puedes añadir f1/recall con el mismo evaluator)
val evalAcc = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

println("[MLPC] accuracy = " + evalAcc.evaluate(pred))
