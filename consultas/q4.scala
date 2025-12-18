import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(200).setMaxDepth(10).setSeed(42)

val rfModel = rf.fit(train)
val predRF = rfModel.transform(test)

val evalAccRF = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")

println("[RF] accuracy = " + evalAccRF.evaluate(predRF))
