import jdk.nashorn.internal.objects.annotations.Where
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.dsl.expressions.{DslAttr, StringToAttributeConversionHelper}
import org.apache.spark.sql.catalyst.expressions.aggregate.Count
import org.apache.spark.sql.functions.{col, isnull, substring, when}
import org.apache.spark.sql.types.DecimalType


/*
Topic: Supervised Learning
There is a lack of publicly available datasets on financial services and especially in the emerging mobile money
transactions domain. Financial datasets are important for researchers to detect fraud in the system. Let us assume that
Paysim is a financial mobile money simulator designed to detect fraud.

Tasks: Now, with the data pipeline ready, you are required to develop the model and predict fraud using spark streaming.
Find out money transfer only which is greater than 200,000

Data Source: https://www.kaggle.com/ntnu-testimon/paysim1
 */

/*
root
 |-- step: integer (nullable = true)              - Unit of time 1 means 1 hour
 |-- type: string (nullable = true)               - CASH_IN, CASH_OUT, DEBIT, PAYMENT, and TRANSFER
 |-- amount: double (nullable = true)             - amount of the transaction in local currency
 |-- nameOrig: string (nullable = true)           - customer who started the transaction
 |-- oldbalanceOrg: double (nullable = true)      - initial balance before the transaction
 |-- newbalanceOrig: double (nullable = true)     - customer's balance after the transaction
 |-- nameDest: string (nullable = true)           - recipient ID of the transaction.
 |-- oldbalanceDest: double (nullable = true)     - initial recipient balance before the transaction
 |-- newbalanceDest: double (nullable = true)     - recipient's balance after the transaction
 |-- isFraud: integer (nullable = true)           - identifies a fraudulent transaction (1) and non-fraudulent (0)
 |-- isFlaggedFraud: integer (nullable = true)    - flags illegal attempts to transfer more than 200,000 in a single transaction.
 */






object money_tx_fruad_lr {



  def main(args: Array[String]): Unit = {


    val spark = SparkSession
      .builder()
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    val dataspath = "D:\\data_files\\mob_money.csv"

    val first_Mob_DF = spark.read.option("header", "true")
      .option("inferSchema", true)
      .option("mode", "DROPMALFORMED")
      .csv(dataspath)
    first_Mob_DF.printSchema()
    println("Check and count null values present in amount column..")
    println( first_Mob_DF.filter(col("amount").isNull || col("amount")===" ").count())
    println("Check and count null values present in oldbalanceOrg column..")
    println( first_Mob_DF.filter(col("oldbalanceOrg").isNull || col("oldbalanceOrg")===" ").count())
    println("Check and count null values present in newbalanceOrig column..")
    println( first_Mob_DF.filter(col("newbalanceOrig").isNull || col("newbalanceOrig")===" ").count())
    println("Check and count null values present in oldbalanceDest column..")
    println( first_Mob_DF.filter(col("oldbalanceDest").isNull || col("oldbalanceDest")===" ").count())
    println("Check and count null values present in newbalanceDest column..")
    println( first_Mob_DF.filter(col("newbalanceDest").isNull || col("newbalanceDest")===" ").count())
    println("Orginal dataframe first_Mob_Df")
    first_Mob_DF.show(5,false)
/*
+----+--------+--------+-----------+-------------+--------------+-----------+--------------+--------------+-------+--------------+
|step|type    |amount  |nameOrig   |oldbalanceOrg|newbalanceOrig|nameDest   |oldbalanceDest|newbalanceDest|isFraud|isFlaggedFraud|
+----+--------+--------+-----------+-------------+--------------+-----------+--------------+--------------+-------+--------------+
|1   |PAYMENT |9839.64 |C1231006815|170136.0     |160296.36     |M1979787155|0.0           |0.0           |0      |0             |
|1   |PAYMENT |1864.28 |C1666544295|21249.0      |19384.72      |M2044282225|0.0           |0.0           |0      |0             |
|1   |TRANSFER|181.0   |C1305486145|181.0        |0.0           |C553264065 |0.0           |0.0           |1      |0             |
|1   |CASH_OUT|181.0   |C840083671 |181.0        |0.0           |C38997010  |21182.0       |0.0           |1      |0             |
|1   |PAYMENT |11668.14|C2048537720|41554.0      |29885.86      |M1230701703|0.0           |0.0           |0      |0             |
+----+--------+--------+-----------+-------------+--------------+-----------+--------------+--------------+-------+--------------+
 */

   println("Total rows=" + first_Mob_DF.count())

/*
Total rows=6362620
 */
    first_Mob_DF.describe().show()
/*
   +-------+------------------+--------+-----------------+-----------+-----------------+-----------------+-----------+------------------+------------------+--------------------+--------------------+
|summary|              step|    type|           amount|   nameOrig|    oldbalanceOrg|   newbalanceOrig|   nameDest|    oldbalanceDest|    newbalanceDest|             isFraud|      isFlaggedFraud|
+-------+------------------+--------+-----------------+-----------+-----------------+-----------------+-----------+------------------+------------------+--------------------+--------------------+
|  count|           6362620| 6362620|          6362620|    6362620|          6362620|          6362620|    6362620|           6362620|           6362620|             6362620|             6362620|
|   mean|243.39724563151657|    null|179861.9035491319|       null|833883.1040744886|855113.6685785907|       null|1100701.6665196423|1224996.3982019203|0.001290820448180152| 2.51468734577894E-6|
| stddev| 142.3319710491289|    null|603858.2314629349|       null| 2888242.67303756|2924048.502954271|       null| 3399180.112994465|3674128.9421196496|0.035904796801604175|0.001585774705736...|
|    min|                 1| CASH_IN|              0.0|C1000000639|              0.0|              0.0|C1000004082|               0.0|               0.0|                   0|                   0|
|    max|               743|TRANSFER|    9.244551664E7| C999999784|    5.958504037E7|    4.958504037E7| M999999784|    3.5601588935E8|    3.5617927892E8|                   1|                   1|
+-------+------------------+--------+-----------------+-----------+-----------------+-----------------+-----------+------------------+------------------+--------------------+--------------------+
 */
    val exponential_Removed_Df = first_Mob_DF
      .withColumn("amount", col ( "amount").cast(DecimalType(12,2)))
      .withColumn("oldbalanceOrg", col ( "oldbalanceOrg").cast(DecimalType(12,2)))
      .withColumn("newbalanceOrig", col ( "newbalanceOrig").cast(DecimalType(12,2)))
      .withColumn("oldbalanceDest" , col("oldbalanceDest").cast(DecimalType(12,2)))
      .withColumn("newbalanceDest" , col("newbalanceDest").cast(DecimalType(12,2)))

    println("Check and count null values present in amount column..")
    println( exponential_Removed_Df.filter(col("amount").isNull || col("amount")==="").count())
    println("Check and count null values present in oldbalanceOrg column..")
    println( exponential_Removed_Df.filter(col("oldbalanceOrg").isNull || col("oldbalanceOrg")===" ").count())
    println("Check and count null values present in newbalanceOrig column..")
    println( exponential_Removed_Df.filter(col("newbalanceOrig").isNull || col("newbalanceOrig")===" ").count())
    println("Check and count null values present in oldbalanceDest column..")
    println( exponential_Removed_Df.filter(col("oldbalanceDest").isNull || col("oldbalanceDest")===" ").count())
    println("Check and count null values present in newbalanceDest column..")
    println( exponential_Removed_Df.filter(col("newbalanceDest").isNull || col("newbalanceDest")===" ").count())
    exponential_Removed_Df.describe().show()


/*
 +-------+------------------+--------+-----------------+-----------+----------------+-----------------+-----------+------------------+-----------------+--------------------+--------------------+
|summary|              step|    type|           amount|   nameOrig|   oldbalanceOrg|   newbalanceOrig|   nameDest|    oldbalanceDest|   newbalanceDest|             isFraud|      isFlaggedFraud|
+-------+------------------+--------+-----------------+-----------+----------------+-----------------+-----------+------------------+-----------------+--------------------+--------------------+
|  count|           6362620| 6362620|          6362620|    6362620|         6362620|          6362620|    6362620|           6361813|          6361635|             6362620|             6362620|
|   mean|243.39724563151657|    null|    179861.903549|       null|   833883.104074|    855113.668579|       null|    1082143.256370|   1202310.019212|0.001290820448180152| 2.51468734577894E-6|
| stddev| 142.3319710491289|    null|603858.2314629349|       null|2888242.67303756|2924048.502954271|       null|2923773.0607544994|3132674.888778381|0.035904796801604175|0.001585774705736...|
|    min|                 1| CASH_IN|             0.00|C1000000639|            0.00|             0.00|C1000004082|              0.00|             0.00|                   0|                   0|
|    max|               743|TRANSFER|      92445516.64| C999999784|     59585040.37|      49585040.37| M999999784|       99962163.31|      99988267.04|                   1|                   1|
+-------+------------------+--------+-----------------+-----------+----------------+-----------------+-----------+------------------+-----------------+--------------------+--------------------+

 */
//exponential_Removed_Df.createOrReplaceTempView("first_view")
    //spark.sql("select count(*) from first_view where amount > 200000 AND isFlaggedFraud = 1 ").show()
/*
+--------+
|count(1)|
+--------+
|      16|
+--------+
 */
    //spark.sql("select count(*) from first_view where amount > 200000 AND isFraud = 1 ").show()
/*
+--------+
|count(1)|
+--------+
|    5471|
+--------+
 */

    //exponential_Removed_Df.groupBy("type").pivot("isFraud").count().show()
/*
+--------+-------+----+
|    type|      0|   1|
+--------+-------+----+
|TRANSFER| 528812|4097|
| CASH_IN|1399284|null|
|CASH_OUT|2233384|4116|
| PAYMENT|2151495|null|
|   DEBIT|  41432|null|
+--------+-------+----+
 */

    //exponential_Removed_Df.groupBy("type").pivot("isFlaggedFraud").count().show()
/*
+--------+-------+----+
|    type|      0|   1|
+--------+-------+----+
|TRANSFER| 532893|  16|
| CASH_IN|1399284|null|
|CASH_OUT|2237500|null|
| PAYMENT|2151495|null|
|   DEBIT|  41432|null|
+--------+-------+----+

 */

    // Null value checking of each columns
    exponential_Removed_Df


    println("null values in oldbalanceDest")
    exponential_Removed_Df.filter(col("oldbalanceDest").isNull).show(false)
/*
+----+--------+-----------+-----------+-------------+--------------+-----------+--------------+--------------+-------+--------------+
|step|type    |amount     |nameOrig   |oldbalanceOrg|newbalanceOrig|nameDest   |oldbalanceDest|newbalanceDest|isFraud|isFlaggedFraud|
+----+--------+-----------+-----------+-------------+--------------+-----------+--------------+--------------+-------+--------------+
|279 |TRANSFER|46211920.92|C256397271 |0.00         |0.00          |C439737079 |null          |null          |0      |0             |
|280 |CASH_IN |83012.59   |C1256561004|71105.00     |154117.59     |C439737079 |null          |null          |0      |0             |
|281 |TRANSFER|16113881.40|C441984477 |0.00         |0.00          |C1320946922|null          |null          |0      |0             |
|281 |TRANSFER|15960347.69|C718364187 |0.00         |0.00          |C817622325 |null          |null          |0      |0             |
|281 |TRANSFER|40639589.17|C385252041 |0.00         |0.00          |C744189981 |null          |null          |0      |0             |
|281 |TRANSFER|8525617.97 |C786109828 |0.00         |0.00          |C172409641 |null          |null          |0      |0             |
|282 |CASH_IN |21791.57   |C269932098 |828107.45    |849899.02     |C268913927 |null          |null          |0      |0             |
|282 |TRANSFER|53920358.88|C140591783 |0.00         |0.00          |C268913927 |null          |null          |0      |0             |
|282 |TRANSFER|32278941.07|C1363464542|0.00         |0.00          |C707403537 |null          |null          |0      |0             |
|282 |TRANSFER|7658569.85 |C193058025 |0.00         |0.00          |C325534370 |null          |null          |0      |0             |
|282 |TRANSFER|37618204.80|C1551396234|0.00         |0.00          |C172409641 |null          |null          |0      |0             |
|283 |CASH_OUT|430136.14  |C1461274013|0.00         |0.00          |C268913927 |null          |null          |0      |0             |
|283 |CASH_IN |122043.15  |C751477336 |11521.00     |133564.15     |C439737079 |null          |null          |0      |0             |
|283 |TRANSFER|56951424.46|C256698564 |0.00         |0.00          |C439737079 |null          |null          |0      |0             |
|283 |TRANSFER|25107947.96|C960156586 |0.00         |0.00          |C327052591 |null          |null          |0      |0             |
|283 |TRANSFER|22378835.11|C1248306329|0.00         |0.00          |C20253152  |null          |null          |0      |0             |
|283 |TRANSFER|38991659.40|C1971023464|0.00         |0.00          |C707403537 |null          |null          |0      |0             |
|284 |TRANSFER|54561579.68|C173148914 |0.00         |0.00          |C20253152  |null          |null          |0      |0             |
|284 |TRANSFER|401672.87  |C1596298061|0.00         |0.00          |C1602144022|null          |null          |0      |0             |
|284 |TRANSFER|35535682.52|C1551488980|0.00         |0.00          |C707403537 |null          |null          |0      |0             |
+----+--------+-----------+-----------+-------------+--------------+-----------+--------------+--------------+-------+--------------+
 */

    println("null values in newbalanceDest")
    exponential_Removed_Df.filter(col("newbalanceDest").isNull).show(false)
/*
    +----+--------+-----------+-----------+-------------+--------------+-----------+--------------+--------------+-------+--------------+
|step|type    |amount     |nameOrig   |oldbalanceOrg|newbalanceOrig|nameDest   |oldbalanceDest|newbalanceDest|isFraud|isFlaggedFraud|
+----+--------+-----------+-----------+-------------+--------------+-----------+--------------+--------------+-------+--------------+
|274 |TRANSFER|33565429.67|C1532333905|0.00         |0.00          |C817622325 |97014721.15   |null          |0      |0             |
|276 |TRANSFER|58944752.64|C24299338  |0.00         |0.00          |C1320946922|91069898.17   |null          |0      |0             |
|277 |TRANSFER|53957543.97|C1440084225|0.00         |0.00          |C439737079 |92455112.62   |null          |0      |0             |
|278 |TRANSFER|12677638.30|C1687783982|0.00         |0.00          |C243325822 |87429330.64   |null          |0      |0             |
|278 |TRANSFER|60154456.05|C31593462  |0.00         |0.00          |C172409641 |68187825.99   |null          |0      |0             |
|279 |TRANSFER|36946551.76|C1399227772|0.00         |0.00          |C744189981 |64216423.00   |null          |0      |0             |
|279 |TRANSFER|27626977.27|C566758972 |0.00         |0.00          |C947086614 |56468946.16   |null          |0      |0             |
|279 |TRANSFER|41963708.15|C1237771660|0.00         |0.00          |C947086614 |84095923.44   |null          |0      |0             |
|279 |TRANSFER|46211920.92|C256397271 |0.00         |0.00          |C439737079 |null          |null          |0      |0             |
|279 |TRANSFER|49380263.91|C1301835320|0.00         |0.00          |C327052591 |55109167.12   |null          |0      |0             |
|279 |TRANSFER|21058476.50|C1534780687|0.00         |0.00          |C1472140329|33755978.87   |null          |0      |0             |
|279 |TRANSFER|33785599.39|C1587463343|0.00         |0.00          |C1472140329|54814455.37   |null          |0      |0             |
|280 |TRANSFER|24407185.51|C1087167148|0.00         |0.00          |C1472140329|88600054.77   |null          |0      |0             |
|280 |CASH_IN |83012.59   |C1256561004|71105.00     |154117.59     |C439737079 |null          |null          |0      |0             |
|281 |TRANSFER|39725690.20|C1793966542|0.00         |0.00          |C2120613885|82625297.36   |null          |0      |0             |
|281 |TRANSFER|16113881.40|C441984477 |0.00         |0.00          |C1320946922|null          |null          |0      |0             |
|281 |TRANSFER|15960347.69|C718364187 |0.00         |0.00          |C817622325 |null          |null          |0      |0             |
|281 |TRANSFER|52042803.47|C1223028177|0.00         |0.00          |C707403537 |53510364.53   |null          |0      |0             |
|281 |TRANSFER|57436619.46|C1139460122|0.00         |0.00          |C310383504 |71731839.82   |null          |0      |0             |
|281 |TRANSFER|40639589.17|C385252041 |0.00         |0.00          |C744189981 |null          |null          |0      |0             |
+----+--------+-----------+-----------+-------------+--------------+-----------+--------------+--------------+-------+--------------+
 */



    //val null_value_removed = exponential_Removed_Df.withColumn("oldbalanceDest", when($"oldbalanceDest".isNull,(0)).otherwise(1))

// remove and replace null value
/* val removed_null_value_df = exponential_Removed_Df
          .na.fill(0,Array("oldbalanceDest"))
          .na.fill(0,Array("newbalanceDest"))
*/

    exponential_Removed_Df.createOrReplaceTempView("first_view")

    //df.na.fill(0)
   val data_set_mob= spark.sql("select amount,type,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,isFraud from first_view   ")
   println("data_set_mob ")
   data_set_mob.show(20,false)

/*
  +---------+--------+-------------+--------------+--------------+--------------+-------+
|amount   |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|isFraud|
+---------+--------+-------------+--------------+--------------+--------------+-------+
|9839.64  |PAYMENT |170136.00    |160296.36     |0.00          |0.00          |0      |
|1864.28  |PAYMENT |21249.00     |19384.72      |0.00          |0.00          |0      |
|181.00   |TRANSFER|181.00       |0.00          |0.00          |0.00          |1      |
|181.00   |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |1      |
|11668.14 |PAYMENT |41554.00     |29885.86      |0.00          |0.00          |0      |
|7817.71  |PAYMENT |53860.00     |46042.29      |0.00          |0.00          |0      |
|7107.77  |PAYMENT |183195.00    |176087.23     |0.00          |0.00          |0      |
|7861.64  |PAYMENT |176087.23    |168225.59     |0.00          |0.00          |0      |
|4024.36  |PAYMENT |2671.00      |0.00          |0.00          |0.00          |0      |
|5337.77  |DEBIT   |41720.00     |36382.23      |41898.00      |40348.79      |0      |
|9644.94  |DEBIT   |4465.00      |0.00          |10845.00      |157982.12     |0      |
|3099.97  |PAYMENT |20771.00     |17671.03      |0.00          |0.00          |0      |
|2560.74  |PAYMENT |5070.00      |2509.26       |0.00          |0.00          |0      |
|11633.76 |PAYMENT |10127.00     |0.00          |0.00          |0.00          |0      |
|4098.78  |PAYMENT |503264.00    |499165.22     |0.00          |0.00          |0      |
|229133.94|CASH_OUT|15325.00     |0.00          |5083.00       |51513.44      |0      |
|1563.82  |PAYMENT |450.00       |0.00          |0.00          |0.00          |0      |
|1157.86  |PAYMENT |21156.00     |19998.14      |0.00          |0.00          |0      |
|671.64   |PAYMENT |15123.00     |14451.36      |0.00          |0.00          |0      |
|215310.30|TRANSFER|705.00       |0.00          |22425.00      |0.00          |0      |
+---------+--------+-------------+--------------+--------------+--------------+-------+
 */


    data_set_mob.describe().show()
/*
+-------+-----------------+--------+----------------+-----------------+------------------+-----------------+--------------------+
|summary|           amount|    type|   oldbalanceOrg|   newbalanceOrig|    oldbalanceDest|   newbalanceDest|             isFraud|
+-------+-----------------+--------+----------------+-----------------+------------------+-----------------+--------------------+
|  count|          6362620| 6362620|         6362620|          6362620|           6361813|          6361635|             6362620|
|   mean|    179861.903549|    null|   833883.104074|    855113.668579|    1082143.256370|   1202310.019212|0.001290820448180152|
| stddev|603858.2314629349|    null|2888242.67303756|2924048.502954271|2923773.0607544994|3132674.888778381|0.035904796801604175|
|    min|             0.00| CASH_IN|            0.00|             0.00|              0.00|             0.00|                   0|
|    max|      92445516.64|TRANSFER|     59585040.37|      49585040.37|       99962163.31|      99988267.04|                   1|
+-------+-----------------+--------+----------------+-----------------+------------------+-----------------+--------------------+

 */
val indexer = new StringIndexer()
  .setInputCol("type")
  .setOutputCol("type_label")

    val type_Label = indexer
      .setHandleInvalid("keep")
      .fit(data_set_mob)
      .transform(data_set_mob)
    println("type_label printSchema")
    type_Label.printSchema()
/*
root
 |-- amount: decimal(10,2) (nullable = true)
 |-- type: string (nullable = true)
 |-- oldbalanceOrg: decimal(10,2) (nullable = true)
 |-- newbalanceOrig: decimal(10,2) (nullable = true)
 |-- oldbalanceDest: decimal(10,2) (nullable = true)
 |-- newbalanceDest: decimal(10,2) (nullable = true)
 |-- isFraud: integer (nullable = true)
 |-- type_label: double (nullable = false)
 */
    type_Label.show(5,false)

/*
+--------+--------+-------------+--------------+--------------+--------------+-------+----------+
|amount  |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|isFraud|type_label|
+--------+--------+-------------+--------------+--------------+--------------+-------+----------+
|9839.64 |PAYMENT |170136.00    |160296.36     |0.00          |0.00          |0      |1.0       |
|1864.28 |PAYMENT |21249.00     |19384.72      |0.00          |0.00          |0      |1.0       |
|181.00  |TRANSFER|181.00       |0.00          |0.00          |0.00          |1      |3.0       |
|181.00  |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |1      |0.0       |
|11668.14|PAYMENT |41554.00     |29885.86      |0.00          |0.00          |0      |1.0       |
+--------+--------+-------------+--------------+--------------+--------------+-------+----------+
 */
val indexer_2 = new StringIndexer()
  .setInputCol("isFraud")
  .setOutputCol("label")

    val isfraud_Label = indexer_2
      .setHandleInvalid("keep")
      .fit(type_Label)
      .transform(type_Label)
    println("isfraud_label printSchema")
    isfraud_Label.printSchema()
/*
  root
 |-- amount: decimal(10,2) (nullable = true)
 |-- type: string (nullable = true)
 |-- oldbalanceOrg: decimal(10,2) (nullable = true)
 |-- newbalanceOrig: decimal(10,2) (nullable = true)
 |-- oldbalanceDest: decimal(10,2) (nullable = true)
 |-- newbalanceDest: decimal(10,2) (nullable = true)
 |-- isFraud: integer (nullable = true)
 |-- type_label: double (nullable = false)
 |-- label: double (nullable = false)
 */
    println("isfraud_label final dataframe")
    isfraud_Label.show(10,false)
/*
+--------+--------+-------------+--------------+--------------+--------------+-------+----------+-----+
|amount  |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|isFraud|type_label|label|
+--------+--------+-------------+--------------+--------------+--------------+-------+----------+-----+
|9839.64 |PAYMENT |170136.00    |160296.36     |0.00          |0.00          |0      |1.0       |0.0  |
|1864.28 |PAYMENT |21249.00     |19384.72      |0.00          |0.00          |0      |1.0       |0.0  |
|181.00  |TRANSFER|181.00       |0.00          |0.00          |0.00          |1      |3.0       |1.0  |
|181.00  |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |1      |0.0       |1.0  |
|11668.14|PAYMENT |41554.00     |29885.86      |0.00          |0.00          |0      |1.0       |0.0  |
|7817.71 |PAYMENT |53860.00     |46042.29      |0.00          |0.00          |0      |1.0       |0.0  |
|7107.77 |PAYMENT |183195.00    |176087.23     |0.00          |0.00          |0      |1.0       |0.0  |
|7861.64 |PAYMENT |176087.23    |168225.59     |0.00          |0.00          |0      |1.0       |0.0  |
|4024.36 |PAYMENT |2671.00      |0.00          |0.00          |0.00          |0      |1.0       |0.0  |
|5337.77 |DEBIT   |41720.00     |36382.23      |41898.00      |40348.79      |0      |4.0       |0.0  |
+--------+--------+-------------+--------------+--------------+--------------+-------+----------+-----+
 */



val cols = Array("amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","type_label")
    val my_Assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")
    val fruad_Feature = my_Assembler.transform(isfraud_Label)
    println("fruad_Feature Schema")
    fruad_Feature.printSchema()
/*
root
 |-- amount: decimal(10,2) (nullable = true)
 |-- type: string (nullable = true)
 |-- oldbalanceOrg: decimal(10,2) (nullable = true)
 |-- newbalanceOrig: decimal(10,2) (nullable = true)
 |-- oldbalanceDest: decimal(10,2) (nullable = true)
 |-- newbalanceDest: decimal(10,2) (nullable = true)
 |-- isFraud: integer (nullable = true)
 |-- type_label: double (nullable = false)
 |-- label: double (nullable = false)
 |-- features: vector (nullable = true)

 */
    fruad_Feature.createOrReplaceTempView("main_data_view")
    println("fruad_Feature data frame")
    fruad_Feature.show(5,false)
/*
+--------+--------+-------------+--------------+--------------+--------------+-------+----------+-----+----------------------------------------+
|amount  |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|isFraud|type_label|label|features                                |
+--------+--------+-------------+--------------+--------------+--------------+-------+----------+-----+----------------------------------------+
|9839.64 |PAYMENT |170136.00    |160296.36     |0.00          |0.00          |0      |1.0       |0.0  |[9839.64,170136.0,160296.36,0.0,00].0,1.|
|1864.28 |PAYMENT |21249.00     |19384.72      |0.00          |0.00          |0      |1.0       |0.0  |[1864.28,21249.0,19384.72,0.0,0.0,1.0]  |
|181.00  |TRANSFER|181.00       |0.00          |0.00          |0.00          |1      |3.0       |1.0  |[181.0,181.0,0.0,0.0,0.0,3.0]           |
|181.00  |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |1      |0.0       |1.0  |[181.0,181.0,0.0,21182.0,0.0,0.0]       |
|11668.14|PAYMENT |41554.00     |29885.86      |0.00          |0.00          |0      |1.0       |0.0  |[11668.14,41554.0,29885.86,0.0,0.0,1.0] |
+--------+--------+-------------+--------------+--------------+--------------+-------+----------+-----+----------------------------------------+
 */

val seed = 5043
    val Array(trainData,testData)=fruad_Feature.randomSplit(Array(0.7,0.3),seed)

    val logisticRegression=new LogisticRegression()
      .setMaxIter(100)
      .setRegParam(0.02)
      .setElasticNetParam(0.8)

    val logisticRegression_model=logisticRegression.fit(trainData)

    val predictionDf=logisticRegression_model.transform(testData)
    println("predictionDF dataframe")
    predictionDf.show(5,false)
/*
+------+--------+-------------+--------------+--------------+--------------+-------+----------+-----+----------------------------------------+----------------------------------------------------------+----------------------------------------------------------------+----------+
|amount|type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|isFraud|type_label|label|features                                |rawPrediction                                             |probability                                                     |prediction|
+------+--------+-------------+--------------+--------------+--------------+-------+----------+-----+----------------------------------------+----------------------------------------------------------+----------------------------------------------------------------+----------+
|0.63  |PAYMENT |1256931.38   |1256930.74    |0.00          |0.00          |0      |1.0       |0.0  |[0.63,1256931.38,1256930.74,0.0,0.0,1.0]|[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|1.33  |PAYMENT |0.00         |0.00          |0.00          |0.00          |0      |1.0       |0.0  |(6,[0,5],[1.33,1.0])                    |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|1.58  |CASH_OUT|0.00         |0.00          |197938.48     |70090.20      |0      |0.0       |0.0  |[1.58,0.0,0.0,197938.48,70090.2,0.0]    |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|1.83  |PAYMENT |45254.28     |45252.45      |0.00          |0.00          |0      |1.0       |0.0  |[1.83,45254.28,45252.45,0.0,0.0,1.0]    |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|1.98  |PAYMENT |329956.56    |329954.58     |0.00          |0.00          |0      |1.0       |0.0  |[1.98,329956.56,329954.58,0.0,0.0,1.0]  |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
+------+--------+-------------+--------------+--------------+--------------+-------+----------+-----+----------------------------------------+----------------------------------------------------------+----------------------------------------------------------------+----------+
 */
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
    val accuracy = evaluator.evaluate(predictionDf)
    println("accuracy % = " + accuracy * 100 )
/*
    accuracy % = 99.8083243196308
 */

    val df1 = spark.sql( "select amount, type, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest from first_view where type ='TRANSFER' OR type = 'CASH_OUT' AND amount > 1000000 " )
    println("df1 details...")
    df1.describe().show()


/*
+-------+------------------+--------+------------------+------------------+------------------+-----------------+
|summary|            amount|    type|     oldbalanceOrg|    newbalanceOrig|    oldbalanceDest|   newbalanceDest|
+-------+------------------+--------+------------------+------------------+------------------+-----------------+
|  count|               500|     500|               500|               500|               500|              500|
|   mean|      143360.79712|    null|1040295.2508999996|1071293.7867199995| 602956.0896199998|     1243080.7127|
| stddev|292009.89894045657|    null| 2226301.040381144|2278118.9148399737|1730418.4187416106|3658710.094330481|
|    min|              8.73| CASH_IN|               0.0|               0.0|               0.0|              0.0|
|    max|        2545478.01|TRANSFER|        8623151.43|        8674805.32|             1.7E7|           1.92E7|
+-------+------------------+--------+------------------+------------------+------------------+-----------------+
 */

    println("df1 show...")
    df1.show(10,false)
/*
+--------+--------+-------------+--------------+--------------+--------------+
|amount  |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|
+--------+--------+-------------+--------------+--------------+--------------+
|9839.64 |PAYMENT |170136.0     |160296.36     |0.0           |0.0           |
|1864.28 |PAYMENT |21249.0      |19384.72      |0.0           |0.0           |
|181.0   |TRANSFER|181.0        |0.0           |0.0           |0.0           |
|181.0   |CASH_OUT|181.0        |0.0           |21182.0       |0.0           |
|11668.14|PAYMENT |41554.0      |29885.86      |0.0           |0.0           |
|7817.71 |PAYMENT |53860.0      |46042.29      |0.0           |0.0           |
|7107.77 |PAYMENT |183195.0     |176087.23     |0.0           |0.0           |
|7861.64 |PAYMENT |176087.23    |168225.59     |0.0           |0.0           |
|4024.36 |PAYMENT |2671.0       |0.0           |0.0           |0.0           |
|5337.77 |DEBIT   |41720.0      |36382.23      |41898.0       |40348.79      |
+--------+--------+-------------+--------------+--------------+--------------+
 */

val exp_rem_df1= df1
  .withColumn("amount", col ( "amount").cast(DecimalType(10,2)))
  .withColumn("oldbalanceOrg", col ( "oldbalanceOrg").cast(DecimalType(12,2)))
  .withColumn("newbalanceOrig", col ( "newbalanceOrig").cast(DecimalType(12,2)))
  .withColumn("oldbalanceDest" , col("oldbalanceDest").cast(DecimalType(12,2)))
  .withColumn("newbalanceDest" , col("newbalanceDest").cast(DecimalType(12,2)))
    exp_rem_df1.describe().show()
    println("exp_rem_df1_describe")
/*
+-------+------------------+--------+-----------------+------------------+------------------+-----------------+
|summary|            amount|    type|    oldbalanceOrg|    newbalanceOrig|    oldbalanceDest|   newbalanceDest|
+-------+------------------+--------+-----------------+------------------+------------------+-----------------+
|  count|               500|     500|              500|               500|               500|              500|
|   mean|     143360.797120|    null|   1040295.250900|    1071293.786720|     602956.089620|   1243080.712700|
| stddev|292009.89894045657|    null|2226301.040381144|2278118.9148399737|1730418.4187416106|3658710.094330481|
|    min|              8.73| CASH_IN|             0.00|              0.00|              0.00|             0.00|
|    max|        2545478.01|TRANSFER|       8623151.43|        8674805.32|       17000000.00|      19200000.00|
+-------+------------------+--------+-----------------+------------------+------------------+-----------------+
 */


    exp_rem_df1.createOrReplaceTempView("df1_view")
    val dataset_final= spark.sql("select amount,type,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest from df1_view   ")
    println("dataset final details")
    dataset_final.describe().show()
/*
+-------+------------------+--------+-----------------+------------------+------------------+-----------------+
|summary|            amount|    type|    oldbalanceOrg|    newbalanceOrig|    oldbalanceDest|   newbalanceDest|
+-------+------------------+--------+-----------------+------------------+------------------+-----------------+
|  count|               500|     500|              500|               500|               500|              500|
|   mean|     143360.797120|    null|   1040295.250900|    1071293.786720|     602956.089620|   1243080.712700|
| stddev|292009.89894045657|    null|2226301.040381144|2278118.9148399737|1730418.4187416106|3658710.094330481|
|    min|              8.73| CASH_IN|             0.00|              0.00|              0.00|             0.00|
|    max|        2545478.01|TRANSFER|       8623151.43|        8674805.32|       17000000.00|      19200000.00|
+-------+------------------+--------+-----------------+------------------+------------------+-----------------+

 */
    println("dataset final")
    dataset_final.show(20,false)
/*
+---------+--------+-------------+--------------+--------------+--------------+
|amount   |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|
+---------+--------+-------------+--------------+--------------+--------------+
|9839.64  |PAYMENT |170136.00    |160296.36     |0.00          |0.00          |
|1864.28  |PAYMENT |21249.00     |19384.72      |0.00          |0.00          |
|181.00   |TRANSFER|181.00       |0.00          |0.00          |0.00          |
|181.00   |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |
|11668.14 |PAYMENT |41554.00     |29885.86      |0.00          |0.00          |
|7817.71  |PAYMENT |53860.00     |46042.29      |0.00          |0.00          |
|7107.77  |PAYMENT |183195.00    |176087.23     |0.00          |0.00          |
|7861.64  |PAYMENT |176087.23    |168225.59     |0.00          |0.00          |
|4024.36  |PAYMENT |2671.00      |0.00          |0.00          |0.00          |
|5337.77  |DEBIT   |41720.00     |36382.23      |41898.00      |40348.79      |
|9644.94  |DEBIT   |4465.00      |0.00          |10845.00      |157982.12     |
|3099.97  |PAYMENT |20771.00     |17671.03      |0.00          |0.00          |
|2560.74  |PAYMENT |5070.00      |2509.26       |0.00          |0.00          |
|11633.76 |PAYMENT |10127.00     |0.00          |0.00          |0.00          |
|4098.78  |PAYMENT |503264.00    |499165.22     |0.00          |0.00          |
|229133.94|CASH_OUT|15325.00     |0.00          |5083.00       |51513.44      |
|1563.82  |PAYMENT |450.00       |0.00          |0.00          |0.00          |
|1157.86  |PAYMENT |21156.00     |19998.14      |0.00          |0.00          |
|671.64   |PAYMENT |15123.00     |14451.36      |0.00          |0.00          |
|215310.30|TRANSFER|705.00       |0.00          |22425.00      |0.00          |
+---------+--------+-------------+--------------+--------------+--------------+
 */
val type_Label_new = indexer
  .setHandleInvalid("keep")
  .fit(dataset_final)
  .transform(dataset_final)
    println("type_label_new printSchema")
    type_Label_new.printSchema()
/*
  root
 |-- amount: decimal(10,2) (nullable = true)
 |-- type: string (nullable = true)
 |-- oldbalanceOrg: decimal(10,2) (nullable = true)
 |-- newbalanceOrig: decimal(10,2) (nullable = true)
 |-- oldbalanceDest: decimal(10,2) (nullable = true)
 |-- newbalanceDest: decimal(10,2) (nullable = true)
 |-- type_label: double (nullable = false)


 */
  type_Label_new.show(10,false)
    println("type_Label_new")
/*
+--------+--------+-------------+--------------+--------------+--------------+----------+
|amount  |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|type_label|
+--------+--------+-------------+--------------+--------------+--------------+----------+
|9839.64 |PAYMENT |170136.00    |160296.36     |0.00          |0.00          |0.0       |
|1864.28 |PAYMENT |21249.00     |19384.72      |0.00          |0.00          |0.0       |
|181.00  |TRANSFER|181.00       |0.00          |0.00          |0.00          |3.0       |
|181.00  |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |1.0       |
|11668.14|PAYMENT |41554.00     |29885.86      |0.00          |0.00          |0.0       |
|7817.71 |PAYMENT |53860.00     |46042.29      |0.00          |0.00          |0.0       |
|7107.77 |PAYMENT |183195.00    |176087.23     |0.00          |0.00          |0.0       |
|7861.64 |PAYMENT |176087.23    |168225.59     |0.00          |0.00          |0.0       |
|4024.36 |PAYMENT |2671.00      |0.00          |0.00          |0.00          |0.0       |
|5337.77 |DEBIT   |41720.00     |36382.23      |41898.00      |40348.79      |4.0       |
+--------+--------+-------------+--------------+--------------+--------------+----------+
 */
val cols_new = Array("amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","type_label")
    val my_Assembler_new = new VectorAssembler()
      .setInputCols(cols_new)
      .setOutputCol("features")
    val final_new_data_Feature = my_Assembler_new.transform(type_Label_new)
    println("final_new_data_Feature Schema")
    final_new_data_Feature.printSchema()
/*
root
 |-- amount: decimal(10,2) (nullable = true)
 |-- type: string (nullable = true)
 |-- oldbalanceOrg: decimal(10,2) (nullable = true)
 |-- newbalanceOrig: decimal(10,2) (nullable = true)
 |-- oldbalanceDest: decimal(10,2) (nullable = true)
 |-- newbalanceDest: decimal(10,2) (nullable = true)
 |-- type_label: double (nullable = false)
 |-- features: vector (nullable = true)
 */
    println("final_new_data_Feature ")
    final_new_data_Feature.show(10,false)
/*
+--------+--------+-------------+--------------+--------------+--------------+----------+-----------------------------------------------+
|amount  |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|type_label|features                                       |
+--------+--------+-------------+--------------+--------------+--------------+----------+-----------------------------------------------+
|9839.64 |PAYMENT |170136.00    |160296.36     |0.00          |0.00          |0.0       |[9839.64,170136.0,160296.36,0.0,0.0,0.0]       |
|1864.28 |PAYMENT |21249.00     |19384.72      |0.00          |0.00          |0.0       |[1864.28,21249.0,19384.72,0.0,0.0,0.0]         |
|181.00  |TRANSFER|181.00       |0.00          |0.00          |0.00          |3.0       |[181.0,181.0,0.0,0.0,0.0,3.0]                  |
|181.00  |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |1.0       |[181.0,181.0,0.0,21182.0,0.0,1.0]              |
|11668.14|PAYMENT |41554.00     |29885.86      |0.00          |0.00          |0.0       |[11668.14,41554.0,29885.86,0.0,0.0,0.0]        |
|7817.71 |PAYMENT |53860.00     |46042.29      |0.00          |0.00          |0.0       |[7817.71,53860.0,46042.29,0.0,0.0,0.0]         |
|7107.77 |PAYMENT |183195.00    |176087.23     |0.00          |0.00          |0.0       |[7107.77,183195.0,176087.23,0.0,0.0,0.0]       |
|7861.64 |PAYMENT |176087.23    |168225.59     |0.00          |0.00          |0.0       |[7861.64,176087.23,168225.59,0.0,0.0,0.0]      |
|4024.36 |PAYMENT |2671.00      |0.00          |0.00          |0.00          |0.0       |(6,[0,1],[4024.36,2671.0])                     |
|5337.77 |DEBIT   |41720.00     |36382.23      |41898.00      |40348.79      |4.0       |[5337.77,41720.0,36382.23,41898.0,40348.79,4.0]|
+--------+--------+-------------+--------------+--------------+--------------+----------+-----------------------------------------------+
 */
    println("logisticRegression_new")
    val df3 = logisticRegression_model.transform(final_new_data_Feature)
    println("regression")
    df3.show(10,false)
/*
+--------+--------+-------------+--------------+--------------+--------------+----------+-----------------------------------------------+----------------------------------------------------------+----------------------------------------------------------------+----------+
|amount  |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|type_label|features                                       |rawPrediction                                             |probability                                                     |prediction|
+--------+--------+-------------+--------------+--------------+--------------+----------+-----------------------------------------------+----------------------------------------------------------+----------------------------------------------------------------+----------+
|9839.64 |PAYMENT |170136.00    |160296.36     |0.00          |0.00          |0.0       |[9839.64,170136.0,160296.36,0.0,0.0,0.0]       |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|1864.28 |PAYMENT |21249.00     |19384.72      |0.00          |0.00          |0.0       |[1864.28,21249.0,19384.72,0.0,0.0,0.0]         |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|181.00  |TRANSFER|181.00       |0.00          |0.00          |0.00          |3.0       |[181.0,181.0,0.0,0.0,0.0,3.0]                  |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|181.00  |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |1.0       |[181.0,181.0,0.0,21182.0,0.0,1.0]              |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|11668.14|PAYMENT |41554.00     |29885.86      |0.00          |0.00          |0.0       |[11668.14,41554.0,29885.86,0.0,0.0,0.0]        |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|7817.71 |PAYMENT |53860.00     |46042.29      |0.00          |0.00          |0.0       |[7817.71,53860.0,46042.29,0.0,0.0,0.0]         |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|7107.77 |PAYMENT |183195.00    |176087.23     |0.00          |0.00          |0.0       |[7107.77,183195.0,176087.23,0.0,0.0,0.0]       |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|7861.64 |PAYMENT |176087.23    |168225.59     |0.00          |0.00          |0.0       |[7861.64,176087.23,168225.59,0.0,0.0,0.0]      |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|4024.36 |PAYMENT |2671.00      |0.00          |0.00          |0.00          |0.0       |(6,[0,1],[4024.36,2671.0])                     |[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
|5337.77 |DEBIT   |41720.00     |36382.23      |41898.00      |40348.79      |4.0       |[5337.77,41720.0,36382.23,41898.0,40348.79,4.0]|[9.733103840160469,3.0861342864919012,-12.819238126652369]|[0.9987037328111104,0.0012962670285359148,1.603537065181099E-10]|0.0       |
+--------+--------+-------------+--------------+--------------+--------------+----------+-----------------------------------------------+----------------------------------------------------------+----------------------------------------------------------------+----------+


 */
    df3.createOrReplaceTempView("df3")
    val df4 = spark.sql("select amount,type,type_label,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,prediction from df3")
    println("new_data_set_final")
    df4.show(10,false)
/*
+--------+--------+----------+-------------+--------------+--------------+--------------+----------+
|amount  |type    |type_label|oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|prediction|
+--------+--------+----------+-------------+--------------+--------------+--------------+----------+
|9839.64 |PAYMENT |0.0       |170136.00    |160296.36     |0.00          |0.00          |0.0       |
|1864.28 |PAYMENT |0.0       |21249.00     |19384.72      |0.00          |0.00          |0.0       |
|181.00  |TRANSFER|3.0       |181.00       |0.00          |0.00          |0.00          |0.0       |
|181.00  |CASH_OUT|1.0       |181.00       |0.00          |21182.00      |0.00          |0.0       |
|11668.14|PAYMENT |0.0       |41554.00     |29885.86      |0.00          |0.00          |0.0       |
|7817.71 |PAYMENT |0.0       |53860.00     |46042.29      |0.00          |0.00          |0.0       |
|7107.77 |PAYMENT |0.0       |183195.00    |176087.23     |0.00          |0.00          |0.0       |
|7861.64 |PAYMENT |0.0       |176087.23    |168225.59     |0.00          |0.00          |0.0       |
|4024.36 |PAYMENT |0.0       |2671.00      |0.00          |0.00          |0.00          |0.0       |
|5337.77 |DEBIT   |4.0       |41720.00     |36382.23      |41898.00      |40348.79      |0.0       |
+--------+--------+----------+-------------+--------------+--------------+--------------+----------+
 */
    df4.createOrReplaceTempView("new_dataset_view")
    println("df4 describe")
    df4.describe().show()
/*
+-------+------------------+--------+------------------+-----------------+------------------+------------------+-----------------+----------+
|summary|            amount|    type|        type_label|    oldbalanceOrg|    newbalanceOrig|    oldbalanceDest|   newbalanceDest|prediction|
+-------+------------------+--------+------------------+-----------------+------------------+------------------+-----------------+----------+
|  count|               500|     500|               500|              500|               500|               500|              500|       500|
|   mean|     143360.797120|    null|             1.246|   1040295.250900|    1071293.786720|     602956.089620|   1243080.712700|       0.0|
| stddev|292009.89894045657|    null|1.2200249677132373|2226301.040381144|2278118.9148399737|1730418.4187416106|3658710.094330481|       0.0|
|    min|              8.73| CASH_IN|               0.0|             0.00|              0.00|              0.00|             0.00|       0.0|
|    max|        2545478.01|TRANSFER|               4.0|       8623151.43|        8674805.32|       17000000.00|      19200000.00|       0.0|
+-------+------------------+--------+------------------+-----------------+------------------+------------------+-----------------+----------+
 */


    val final_out = spark.sql("SELECT first_view.amount,first_view.type,first_view.oldbalanceOrg," +
      "first_view.newbalanceOrig,first_view.oldbalanceDest,first_view.newbalanceDest,new_dataset_view.prediction," +
      " first_view.isFraud  FROM first_view  " +
      "JOIN new_dataset_view ON first_view.amount = new_dataset_view.amount AND first_view.type = new_dataset_view.type " +
      "AND  first_view.oldbalanceOrg = new_dataset_view.oldbalanceOrg AND " +
      "first_view.newbalanceOrig = new_dataset_view.newbalanceOrig " +
      "AND first_view.oldbalanceDest = new_dataset_view.oldbalanceDest AND " +
      "first_view.newbalanceDest = new_dataset_view.newbalanceDest" +
      " GROUP BY first_view.amount,first_view.type,first_view.oldbalanceOrg,first_view.newbalanceOrig,first_view.oldbalanceDest," +
      "first_view.newbalanceDest,first_view.isFraud,new_dataset_view.prediction ")


    println("final_out dataframe")
    final_out.show(25,false)

/*
+------+--------+-------------+--------------+--------------+--------------+----------+-------+
|amount|type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|prediction|isFraud|
+------+--------+-------------+--------------+--------------+--------------+----------+-------+
|7.23  |TRANSFER|0.00         |0.00          |426840.94     |426848.17     |0.0       |0      |
|30.27 |TRANSFER|0.00         |0.00          |18997.38      |19027.65      |0.0       |0      |
|142.17|TRANSFER|0.00         |0.00          |4282363.08    |4282505.25    |0.0       |0      |
|161.04|TRANSFER|51708.00     |51546.96      |7447925.00    |7448086.03    |0.0       |0      |
|168.36|TRANSFER|0.00         |0.00          |1659206.45    |1659374.81    |0.0       |0      |
|168.98|TRANSFER|88187.00     |88018.02      |0.00          |168.98        |0.0       |0      |
|170.00|TRANSFER|170.00       |0.00          |0.00          |0.00          |0.0       |1      |
|196.44|TRANSFER|0.00         |0.00          |77029.54      |77225.97      |0.0       |0      |
|202.69|TRANSFER|99078.00     |98875.31      |0.00          |202.69        |0.0       |0      |
|231.37|TRANSFER|0.00         |0.00          |3550226.88    |3550458.25    |0.0       |0      |
|250.78|TRANSFER|0.00         |0.00          |141944.33     |142195.11     |0.0       |0      |
|271.49|TRANSFER|0.00         |0.00          |164982.76     |242817.53     |0.0       |0      |
|274.21|TRANSFER|38042.00     |37767.79      |2640688.89    |2640963.10    |0.0       |0      |
|296.93|TRANSFER|0.00         |0.00          |76589.60      |76886.53      |0.0       |0      |
|328.28|TRANSFER|0.00         |0.00          |4853268.79    |4853597.07    |0.0       |0      |
|330.87|TRANSFER|0.00         |0.00          |163355.32     |163686.19     |0.0       |0      |
|333.80|TRANSFER|35082.00     |34748.20      |3390409.40    |3390743.20    |0.0       |0      |
|389.26|TRANSFER|50043.00     |49653.74      |0.00          |389.26        |0.0       |0      |
|390.92|TRANSFER|0.00         |0.00          |4791117.03    |4791507.95    |0.0       |0      |
|417.04|TRANSFER|0.00         |0.00          |1207857.68    |1147702.89    |0.0       |0      |
|459.38|TRANSFER|459.38       |0.00          |0.00          |0.00          |0.0       |1      |
|464.93|TRANSFER|14522.00     |14057.07      |95576.95      |96041.87      |0.0       |0      |
|464.97|TRANSFER|0.00         |0.00          |2782295.06    |2782760.02    |0.0       |0      |
|471.32|TRANSFER|0.00         |0.00          |342743.05     |343214.37     |0.0       |0      |
|478.81|TRANSFER|0.00         |0.00          |260503.14     |260981.95     |0.0       |0      |
+------+--------+-------------+--------------+--------------+--------------+----------+-------+
 */
    println("prediction filter")
    final_out.filter( "prediction = 1").show(50,false)

    println("filter isfraud")
/*
   +-----------+--------+-------------+--------------+--------------+--------------+----------+-------+
|amount     |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|prediction|isFraud|
+-----------+--------+-------------+--------------+--------------+--------------+----------+-------+
|1161495.82 |TRANSFER|1161495.82   |0.00          |0.00          |0.00          |0.0       |1      |
|262434.54  |TRANSFER|262434.54    |0.00          |0.00          |0.00          |0.0       |1      |
|11308.00   |TRANSFER|11308.00     |0.00          |0.00          |0.00          |0.0       |1      |
|350705.74  |TRANSFER|350705.74    |0.00          |0.00          |1184633.07    |0.0       |1      |
|10224.00   |TRANSFER|10224.00     |0.00          |0.00          |0.00          |0.0       |1      |
|85354.69   |TRANSFER|85354.69     |0.00          |0.00          |0.00          |0.0       |1      |
|22877.00   |TRANSFER|22877.00     |0.00          |0.00          |0.00          |0.0       |1      |
 */
    //final_out.filter( "prediction = 1.0").show(50,false)

    import spark.implicits._
    val df1_man = Seq(
      (205000,"TRANSFER",0,205000,205000,0),
      (1,"CASH_OUT",0,300000,600000,300000),
      (800000,"TRANSFER",500000,0,0,800000),
      (5000,"TRANSFER",5000,0,5000,5000),
      (600000,"CASH_OUT",1000000,400000,1,1),
      (10,"CASH_OUT",300000,299999,0,300000),
      (450000,"TRANSFER",1000000,650000,5000,500000),

    ).toDF("amount","type","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest")
    df1_man.show()
    println("manual data")
/*
+------+--------+-------------+--------------+--------------+--------------+
|amount|    type|oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|
+------+--------+-------------+--------------+--------------+--------------+
|205000|TRANSFER|            0|        205000|        205000|             0|
|     1|CASH_OUT|            0|        300000|        600000|        300000|
|800000|TRANSFER|       500000|             0|             0|        800000|
|  5000|TRANSFER|         5000|             0|          5000|          5000|
|600000|CASH_OUT|      1000000|        400000|             1|             1|
|    10|CASH_OUT|       300000|        299999|             0|        300000|
|450000|TRANSFER|      1000000|        650000|          5000|        500000|
+------+--------+-------------+--------------+--------------+--------------+

 */

    val indexer_man = new StringIndexer()
      .setInputCol("type")
      .setOutputCol("type_label")

    val df1_man_Label = indexer_man
      .setHandleInvalid("keep")
      .fit(df1_man)
      .transform(df1_man)

    df1_man_Label.show()
    println("manual data with type label")
/*
+------+--------+-------------+--------------+--------------+--------------+----------+
|amount|    type|oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|type_label|
+------+--------+-------------+--------------+--------------+--------------+----------+
|205000|TRANSFER|            0|        205000|        205000|             0|       0.0|
|     1|CASH_OUT|            0|        300000|        600000|        300000|       1.0|
|800000|TRANSFER|       500000|             0|             0|        800000|       0.0|
|  5000|TRANSFER|         5000|             0|          5000|          5000|       0.0|
|600000|CASH_OUT|      1000000|        400000|             1|             1|       1.0|
|    10|CASH_OUT|       300000|        299999|             0|        300000|       1.0|
|450000|TRANSFER|      1000000|        650000|          5000|        500000|       0.0|
+------+--------+-------------+--------------+--------------+--------------+----------+
 */

    val cols_man = Array("amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","type_label")
    val my_Assembler_man = new VectorAssembler()
      .setInputCols(cols_man)
      .setOutputCol("features")
    val df2_man = my_Assembler_man.transform(df1_man_Label)

    df2_man.printSchema()
/*
root
 |-- amount: integer (nullable = false)
 |-- type: string (nullable = true)
 |-- oldbalanceOrg: integer (nullable = false)
 |-- newbalanceOrig: integer (nullable = false)
 |-- oldbalanceDest: integer (nullable = false)
 |-- newbalanceDest: integer (nullable = false)
 |-- type_label: double (nullable = false)
 |-- features: vector (nullable = true)

 */

    df2_man.show()
/*
+------+--------+-------------+--------------+--------------+--------------+----------+--------------------+
|amount|    type|oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|type_label|            features|
+------+--------+-------------+--------------+--------------+--------------+----------+--------------------+
|205000|TRANSFER|            0|        205000|        205000|             0|       0.0|[205000.0,0.0,205...|
|     1|CASH_OUT|            0|        300000|        600000|        300000|       1.0|[1.0,0.0,300000.0...|
|800000|TRANSFER|       500000|             0|             0|        800000|       0.0|[800000.0,500000....|
|  5000|TRANSFER|         5000|             0|          5000|          5000|       0.0|[5000.0,5000.0,0....|
|600000|CASH_OUT|      1000000|        400000|             1|             1|       1.0|[600000.0,1000000...|
|    10|CASH_OUT|       300000|        299999|             0|        300000|       1.0|[10.0,300000.0,29...|
|450000|TRANSFER|      1000000|        650000|          5000|        500000|       0.0|[450000.0,1000000...|
+------+--------+-------------+--------------+--------------+--------------+----------+--------------------+
 */

    val df3_man = logisticRegression_model.transform(df2_man)
    df3_man.show()
/*
+------+--------+-------------+--------------+--------------+--------------+----------+--------------------+--------------------+--------------------+----------+
|amount|    type|oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|type_label|            features|       rawPrediction|         probability|prediction|
+------+--------+-------------+--------------+--------------+--------------+----------+--------------------+--------------------+--------------------+----------+
|205000|TRANSFER|            0|        205000|        205000|             0|       0.0|[205000.0,0.0,205...|[9.73310199408213...|[0.99870373281103...|       0.0|
|     1|CASH_OUT|            0|        300000|        600000|        300000|       1.0|[1.0,0.0,300000.0...|[9.73310199408213...|[0.99870373281103...|       0.0|
|800000|TRANSFER|       500000|             0|             0|        800000|       0.0|[800000.0,500000....|[9.73310199408213...|[0.99870373281103...|       0.0|
|  5000|TRANSFER|         5000|             0|          5000|          5000|       0.0|[5000.0,5000.0,0....|[9.73310199408213...|[0.99870373281103...|       0.0|
|600000|CASH_OUT|      1000000|        400000|             1|             1|       1.0|[600000.0,1000000...|[9.73310199408213...|[0.99870373281103...|       0.0|
|    10|CASH_OUT|       300000|        299999|             0|        300000|       1.0|[10.0,300000.0,29...|[9.73310199408213...|[0.99870373281103...|       0.0|
|450000|TRANSFER|      1000000|        650000|          5000|        500000|       0.0|[450000.0,1000000...|[9.73310199408213...|[0.99870373281103...|       0.0|
+------+--------+-------------+--------------+--------------+--------------+----------+--------------------+--------------------+--------------------+----------+

 */

    df3_man.createOrReplaceTempView("view_man")
    val df4_man = spark.sql("select amount,type,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,prediction from view_man")
    df4_man.show()
/*
+------+--------+-------------+--------------+--------------+--------------+----------+
|amount|    type|oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|prediction|
+------+--------+-------------+--------------+--------------+--------------+----------+
|205000|TRANSFER|            0|        205000|        205000|             0|       0.0|
|     1|CASH_OUT|            0|        300000|        600000|        300000|       0.0|
|800000|TRANSFER|       500000|             0|             0|        800000|       0.0|
|  5000|TRANSFER|         5000|             0|          5000|          5000|       0.0|
|600000|CASH_OUT|      1000000|        400000|             1|             1|       0.0|
|    10|CASH_OUT|       300000|        299999|             0|        300000|       0.0|
|450000|TRANSFER|      1000000|        650000|          5000|        500000|       0.0|
+------+--------+-------------+--------------+--------------+--------------+----------+

 */
/*
    val final_out_man = spark.sql("SELECT first_view.amount,first_view.type,first_view.oldbalanceOrg," +
      "first_view.newbalanceOrig,first_view.oldbalanceDest,first_view.newbalanceDest,view_man.prediction," +
      " first_view.isFraud  FROM first_view  " +
      "JOIN view_man ON first_view.amount = view_man.amount AND first_view.type = view_man.type " +
      "AND  first_view.oldbalanceOrg = view_man.oldbalanceOrg AND " +
      "first_view.newbalanceOrig = view_man.newbalanceOrig " +
      "AND first_view.oldbalanceDest = view_man.oldbalanceDest AND " +
      "first_view.newbalanceDest = view_man.newbalanceDest" +
      " GROUP BY first_view.amount,first_view.type,first_view.oldbalanceOrg,first_view.newbalanceOrig,first_view.oldbalanceDest," +
      "first_view.newbalanceDest,first_view.isFraud,view_man.prediction ")
    println("print final_man")
    final_out_man.show()
*/
  }
}
