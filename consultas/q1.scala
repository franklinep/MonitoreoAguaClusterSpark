import org.apache.spark.sql.functions._

val q1 = mediciones
  .groupBy("parametro", "tipo_muestra")
  .agg(
    count("*").as("n_registros"),
    round(avg("resultado"), 6).as("promedio"),
    round(max("resultado"), 6).as("maximo"),
    round(min("resultado"), 6).as("minimo"),
    round(percentile_approx(col("resultado"), lit(0.5), lit(10000)), 6).as("mediana")
  )
  .orderBy(desc("n_registros"))

q1.show(false)
