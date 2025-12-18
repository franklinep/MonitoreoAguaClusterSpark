import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

val base = "hdfs://master:9000/user/master/agua_superficial/json"
val puntos   = spark.read.json(s"$base/dim_puntos.json")
val campanas = spark.read.json(s"$base/dim_campanas.json")
val mediciones = spark.read.json(s"$base/fact_mediciones.json")

println(puntos.count())
println(campanas.count())
println(mediciones.count())
