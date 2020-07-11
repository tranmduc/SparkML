import org.apache.spark.ml.feature.FeatureHasher
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object MyFeatureHasher {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("MyFeatureHasher")
      .config("spark.sql.warehouse.dir", ".")
      .getOrCreate()

    val dataset = spark.createDataFrame(Seq(
      (2.2, true, "1", "foo"),
      (3.3, false, "2", "bar"),
      (4.4, false, "3", "baz"),
      (5.5, false, "4", "foo")
    )).toDF("real", "bool", "stringNum", "string")

    val hasher = new FeatureHasher()
      .setInputCols("real", "bool", "stringNum", "string")
      .setOutputCol("features")

    val featurized = hasher.transform(dataset)
    featurized.show(false)
  }
}
