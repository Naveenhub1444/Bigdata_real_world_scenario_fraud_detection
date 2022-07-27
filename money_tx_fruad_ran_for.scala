import jdk.nashorn.internal.objects.annotations.Where
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
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






object money_tx_fruad_ran_for {



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
    println( first_Mob_DF.filter(col("amount").isNull || col("amount")==="").count())
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

    val classifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(10)
      .setNumTrees(20)
      .setFeatureSubsetStrategy("auto")
      .setSeed(5043)

    val model = classifier.fit(trainData)

    val prediction_df = model.transform(testData)
    println("predictionDF dataframe")
    prediction_df.show(10,false)
    /*
+------+--------+-------------+--------------+--------------+--------------+-------+----------+-----+----------------------------------------+---------------------------------------------+----------------------------------------------+----------+
|amount|type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|isFraud|type_label|label|features                                |rawPrediction                                |probability                                   |prediction|
+------+--------+-------------+--------------+--------------+--------------+-------+----------+-----+----------------------------------------+---------------------------------------------+----------------------------------------------+----------+
|0.63  |PAYMENT |1256931.38   |1256930.74    |0.00          |0.00          |0      |1.0       |0.0  |[0.63,1256931.38,1256930.74,0.0,0.0,1.0]|[19.99916700142441,8.329985755917515E-4,0.0] |[0.9999583500712204,4.164992877958757E-5,0.0] |0.0       |
|1.33  |PAYMENT |0.00         |0.00          |0.00          |0.00          |0      |1.0       |0.0  |(6,[0,5],[1.33,1.0])                    |[19.99640416227334,0.003595837726658073,0.0] |[0.9998202081136671,1.7979188633290366E-4,0.0]|0.0       |
|1.58  |CASH_OUT|0.00         |0.00          |197938.48     |70090.20      |0      |0.0       |0.0  |[1.58,0.0,0.0,197938.48,70090.2,0.0]    |[19.983610539256222,0.016389460743778516,0.0]|[0.9991805269628111,8.194730371889258E-4,0.0] |0.0       |
|1.83  |PAYMENT |45254.28     |45252.45      |0.00          |0.00          |0      |1.0       |0.0  |[1.83,45254.28,45252.45,0.0,0.0,1.0]    |[19.99751672536442,0.0024832746355837863,0.0]|[0.9998758362682209,1.241637317791893E-4,0.0] |0.0       |
|1.98  |PAYMENT |329956.56    |329954.58     |0.00          |0.00          |0      |1.0       |0.0  |[1.98,329956.56,329954.58,0.0,0.0,1.0]  |[19.99916700142441,8.329985755917515E-4,0.0] |[0.9999583500712204,4.164992877958757E-5,0.0] |0.0       |
|2.11  |PAYMENT |0.00         |0.00          |0.00          |0.00          |0      |1.0       |0.0  |(6,[0,5],[2.11,1.0])                    |[19.99640416227334,0.003595837726658073,0.0] |[0.9998202081136671,1.7979188633290366E-4,0.0]|0.0       |
|2.27  |PAYMENT |35999.03     |35996.76      |0.00          |0.00          |0      |1.0       |0.0  |[2.27,35999.03,35996.76,0.0,0.0,1.0]    |[19.997387807426882,0.002612192573120024,0.0]|[0.9998693903713439,1.3060962865600118E-4,0.0]|0.0       |
|2.70  |PAYMENT |22705.00     |22702.30      |0.00          |0.00          |0      |1.0       |0.0  |[2.7,22705.0,22702.3,0.0,0.0,1.0]       |[19.998194740762223,0.0018052592377756,0.0]  |[0.9999097370381111,9.026296188878E-5,0.0]    |0.0       |
|3.05  |PAYMENT |0.00         |0.00          |0.00          |0.00          |0      |1.0       |0.0  |(6,[0,5],[3.05,1.0])                    |[19.99640416227334,0.003595837726658073,0.0] |[0.9998202081136671,1.7979188633290366E-4,0.0]|0.0       |
|3.45  |PAYMENT |100824.64    |100821.19     |0.00          |0.00          |0      |1.0       |0.0  |[3.45,100824.64,100821.19,0.0,0.0,1.0]  |[19.99817984818857,0.0018201518114282045,0.0]|[0.9999089924094285,9.100759057141022E-5,0.0] |0.0       |
+------+--------+-------------+--------------+--------------+--------------+-------+----------+-----+--------------------------------
     */
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
    val accuracy = evaluator.evaluate(prediction_df)
    println("accuracy % = " + accuracy * 100 )
    /*
accuracy % = 99.95617678102035
     */

    val df1 = spark.sql( "select amount, type, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest from first_view where type ='TRANSFER' OR type = 'CASH_OUT'   " )
    println("df1 details...")
    df1.describe().show()


    /*
+-------+-----------------+--------+------------------+-----------------+-----------------+-----------------+
|summary|           amount|    type|     oldbalanceOrg|   newbalanceOrig|   oldbalanceDest|   newbalanceDest|
+-------+-----------------+--------+------------------+-----------------+-----------------+-----------------+
|  count|          2770409| 2770409|           2770409|          2770409|          2770409|          2770409|
|   mean|    317536.140869|    null|      47643.079411|     16091.904679|   1703551.161910|   2049734.436887|
| stddev|887789.6576433469|    null|251325.12749238333|151255.8247602122|4225550.490196659|4676990.061142499|
|    min|             0.00|CASH_OUT|              0.00|             0.00|             0.00|             0.00|
|    max|      92445516.64|TRANSFER|       59585040.37|      49585040.37|     356015889.35|     356179278.92|
+-------+-----------------+--------+------------------+-----------------+-----------------+-----------------+
     */

    println("df1 show...")
    df1.show(10,false)
    /*
+---------+--------+-------------+--------------+--------------+--------------+
|amount   |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|
+---------+--------+-------------+--------------+--------------+--------------+
|181.00   |TRANSFER|181.00       |0.00          |0.00          |0.00          |
|181.00   |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |
|229133.94|CASH_OUT|15325.00     |0.00          |5083.00       |51513.44      |
|215310.30|TRANSFER|705.00       |0.00          |22425.00      |0.00          |
|311685.89|TRANSFER|10835.00     |0.00          |6267.00       |2719172.89    |
|110414.71|CASH_OUT|26845.41     |0.00          |288800.00     |2415.16       |
|56953.90 |CASH_OUT|1942.02      |0.00          |70253.00      |64106.18      |
|5346.89  |CASH_OUT|0.00         |0.00          |652637.00     |6453430.91    |
|23261.30 |CASH_OUT|20411.53     |0.00          |25742.00      |0.00          |
|62610.80 |TRANSFER|79114.00     |16503.20      |517.00        |8383.29       |
+---------+--------+-------------+--------------+--------------+--------------+
     */

    val exp_rem_df1= df1
      .withColumn("amount", col ( "amount").cast(DecimalType(10,2)))
      .withColumn("oldbalanceOrg", col ( "oldbalanceOrg").cast(DecimalType(12,2)))
      .withColumn("newbalanceOrig", col ( "newbalanceOrig").cast(DecimalType(12,2)))
      .withColumn("oldbalanceDest" , col("oldbalanceDest").cast(DecimalType(12,2)))
      .withColumn("newbalanceDest" , col("newbalanceDest").cast(DecimalType(12,2)))
    println("exp_rem_df1_describe")
    exp_rem_df1.describe().show()

    /*
+-------+-----------------+--------+------------------+-----------------+-----------------+-----------------+
|summary|           amount|    type|     oldbalanceOrg|   newbalanceOrig|   oldbalanceDest|   newbalanceDest|
+-------+-----------------+--------+------------------+-----------------+-----------------+-----------------+
|  count|          2770409| 2770409|           2770409|          2770409|          2770409|          2770409|
|   mean|    317536.140869|    null|      47643.079411|     16091.904679|   1703551.161910|   2049734.436887|
| stddev|887789.6576433469|    null|251325.12749238333|151255.8247602122|4225550.490196659|4676990.061142499|
|    min|             0.00|CASH_OUT|              0.00|             0.00|             0.00|             0.00|
|    max|      92445516.64|TRANSFER|       59585040.37|      49585040.37|     356015889.35|     356179278.92|
+-------+-----------------+--------+------------------+-----------------+-----------------+-----------------+
     */


    exp_rem_df1.createOrReplaceTempView("df1_view")
    val dataset_final= spark.sql("select amount,type,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest from df1_view   ")
    println("dataset final details")
    dataset_final.describe().show()
    /*
+-------+-----------------+--------+------------------+-----------------+-----------------+-----------------+
|summary|           amount|    type|     oldbalanceOrg|   newbalanceOrig|   oldbalanceDest|   newbalanceDest|
+-------+-----------------+--------+------------------+-----------------+-----------------+-----------------+
|  count|          2770409| 2770409|           2770409|          2770409|          2770409|          2770409|
|   mean|    317536.140869|    null|      47643.079411|     16091.904679|   1703551.161910|   2049734.436887|
| stddev|887789.6576433469|    null|251325.12749238333|151255.8247602122|4225550.490196659|4676990.061142499|
|    min|             0.00|CASH_OUT|              0.00|             0.00|             0.00|             0.00|
|    max|      92445516.64|TRANSFER|       59585040.37|      49585040.37|     356015889.35|     356179278.92|
+-------+-----------------+--------+------------------+-----------------+-----------------+-----------------+

     */
    println("dataset final")
    dataset_final.show(20,false)
    /*
+---------+--------+-------------+--------------+--------------+--------------+
|amount   |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|
+---------+--------+-------------+--------------+--------------+--------------+
|181.00   |TRANSFER|181.00       |0.00          |0.00          |0.00          |
|181.00   |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |
|229133.94|CASH_OUT|15325.00     |0.00          |5083.00       |51513.44      |
|215310.30|TRANSFER|705.00       |0.00          |22425.00      |0.00          |
|311685.89|TRANSFER|10835.00     |0.00          |6267.00       |2719172.89    |
|110414.71|CASH_OUT|26845.41     |0.00          |288800.00     |2415.16       |
|56953.90 |CASH_OUT|1942.02      |0.00          |70253.00      |64106.18      |
|5346.89  |CASH_OUT|0.00         |0.00          |652637.00     |6453430.91    |
|23261.30 |CASH_OUT|20411.53     |0.00          |25742.00      |0.00          |
|62610.80 |TRANSFER|79114.00     |16503.20      |517.00        |8383.29       |
|82940.31 |CASH_OUT|3017.87      |0.00          |132372.00     |49864.36      |
|47458.86 |CASH_OUT|209534.84    |162075.98     |52120.00      |0.00          |
|136872.92|CASH_OUT|162075.98    |25203.05      |217806.00     |0.00          |
|94253.33 |CASH_OUT|25203.05     |0.00          |99773.00      |965870.05     |
|42712.39 |TRANSFER|10363.39     |0.00          |57901.66      |24044.18      |
|77957.68 |TRANSFER|0.00         |0.00          |94900.00      |22233.65      |
|17231.46 |TRANSFER|0.00         |0.00          |24672.00      |0.00          |
|78766.03 |TRANSFER|0.00         |0.00          |103772.00     |277515.05     |
|224606.64|TRANSFER|0.00         |0.00          |354678.92     |0.00          |
|125872.53|TRANSFER|0.00         |0.00          |348512.00     |3420103.09    |
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
 |-- oldbalanceOrg: decimal(12,2) (nullable = true)
 |-- newbalanceOrig: decimal(12,2) (nullable = true)
 |-- oldbalanceDest: decimal(12,2) (nullable = true)
 |-- newbalanceDest: decimal(12,2) (nullable = true)
 |-- type_label: double (nullable = false)

     */
    type_Label_new.show(10,false)
    println("type_Label_new")
    /*
+---------+--------+-------------+--------------+--------------+--------------+----------+
|amount   |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|type_label|
+---------+--------+-------------+--------------+--------------+--------------+----------+
|181.00   |TRANSFER|181.00       |0.00          |0.00          |0.00          |1.0       |
|181.00   |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |0.0       |
|229133.94|CASH_OUT|15325.00     |0.00          |5083.00       |51513.44      |0.0       |
|215310.30|TRANSFER|705.00       |0.00          |22425.00      |0.00          |1.0       |
|311685.89|TRANSFER|10835.00     |0.00          |6267.00       |2719172.89    |1.0       |
|110414.71|CASH_OUT|26845.41     |0.00          |288800.00     |2415.16       |0.0       |
|56953.90 |CASH_OUT|1942.02      |0.00          |70253.00      |64106.18      |0.0       |
|5346.89  |CASH_OUT|0.00         |0.00          |652637.00     |6453430.91    |0.0       |
|23261.30 |CASH_OUT|20411.53     |0.00          |25742.00      |0.00          |0.0       |
|62610.80 |TRANSFER|79114.00     |16503.20      |517.00        |8383.29       |1.0       |
+---------+--------+-------------+--------------+--------------+--------------+----------+
     */
    val cols_new = Array("amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","type_label")
    val my_Assembler_new = new VectorAssembler()
      .setInputCols(cols_new)
      .setOutputCol("features")
    val final_new_data_Feature = my_Assembler.transform(type_Label_new)
    println("final_new_data_Feature Schema")
    final_new_data_Feature.printSchema()
    /*
oot
 |-- amount: decimal(10,2) (nullable = true)
 |-- type: string (nullable = true)
 |-- oldbalanceOrg: decimal(12,2) (nullable = true)
 |-- newbalanceOrig: decimal(12,2) (nullable = true)
 |-- oldbalanceDest: decimal(12,2) (nullable = true)
 |-- newbalanceDest: decimal(12,2) (nullable = true)
 |-- type_label: double (nullable = false)
 |-- features: vector (nullable = true)
     */
    println("final_new_data_Feature ")
    final_new_data_Feature.show(10,false)
    /*
+---------+--------+-------------+--------------+--------------+--------------+----------+---------------------------------------------+
|amount   |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|type_label|features                                     |
+---------+--------+-------------+--------------+--------------+--------------+----------+---------------------------------------------+
|181.00   |TRANSFER|181.00       |0.00          |0.00          |0.00          |1.0       |[181.0,181.0,0.0,0.0,0.0,1.0]                |
|181.00   |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |0.0       |[181.0,181.0,0.0,21182.0,0.0,0.0]            |
|229133.94|CASH_OUT|15325.00     |0.00          |5083.00       |51513.44      |0.0       |[229133.94,15325.0,0.0,5083.0,51513.44,0.0]  |
|215310.30|TRANSFER|705.00       |0.00          |22425.00      |0.00          |1.0       |[215310.3,705.0,0.0,22425.0,0.0,1.0]         |
|311685.89|TRANSFER|10835.00     |0.00          |6267.00       |2719172.89    |1.0       |[311685.89,10835.0,0.0,6267.0,2719172.89,1.0]|
|110414.71|CASH_OUT|26845.41     |0.00          |288800.00     |2415.16       |0.0       |[110414.71,26845.41,0.0,288800.0,2415.16,0.0]|
|56953.90 |CASH_OUT|1942.02      |0.00          |70253.00      |64106.18      |0.0       |[56953.9,1942.02,0.0,70253.0,64106.18,0.0]   |
|5346.89  |CASH_OUT|0.00         |0.00          |652637.00     |6453430.91    |0.0       |[5346.89,0.0,0.0,652637.0,6453430.91,0.0]    |
|23261.30 |CASH_OUT|20411.53     |0.00          |25742.00      |0.00          |0.0       |[23261.3,20411.53,0.0,25742.0,0.0,0.0]       |
|62610.80 |TRANSFER|79114.00     |16503.20      |517.00        |8383.29       |1.0       |[62610.8,79114.0,16503.2,517.0,8383.29,1.0]  |
+---------+--------+-------------+--------------+--------------+--------------+----------+---------------------------------------------+
     */
    println("random_forest_Regression_new")
    val df3 = model.transform(final_new_data_Feature)
    println("regression")
    df3.show(10,false)

    /*
+---------+--------+-------------+--------------+--------------+--------------+----------+---------------------------------------------+----------------------------------------------+----------------------------------------------+----------+
|amount   |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|type_label|features                                     |rawPrediction                                 |probability                                   |prediction|
+---------+--------+-------------+--------------+--------------+--------------+----------+---------------------------------------------+----------------------------------------------+----------------------------------------------+----------+
|181.00   |TRANSFER|181.00       |0.00          |0.00          |0.00          |1.0       |[181.0,181.0,0.0,0.0,0.0,1.0]                |[19.99640416227334,0.003595837726658073,0.0]  |[0.9998202081136671,1.7979188633290366E-4,0.0]|0.0       |
|181.00   |CASH_OUT|181.00       |0.00          |21182.00      |0.00          |0.0       |[181.0,181.0,0.0,21182.0,0.0,0.0]            |[19.84925737168594,0.15074262831405918,0.0]   |[0.992462868584297,0.007537131415702959,0.0]  |0.0       |
|229133.94|CASH_OUT|15325.00     |0.00          |5083.00       |51513.44      |0.0       |[229133.94,15325.0,0.0,5083.0,51513.44,0.0]  |[19.987018446877823,0.012981553122180106,0.0] |[0.999350922343891,6.490776561090052E-4,0.0]  |0.0       |
|215310.30|TRANSFER|705.00       |0.00          |22425.00      |0.00          |1.0       |[215310.3,705.0,0.0,22425.0,0.0,1.0]         |[19.996233668228083,0.0037663317719205747,0.0]|[0.9998116834114039,1.883165885960287E-4,0.0] |0.0       |
|311685.89|TRANSFER|10835.00     |0.00          |6267.00       |2719172.89    |1.0       |[311685.89,10835.0,0.0,6267.0,2719172.89,1.0]|[19.997089957196604,0.002910042803396845,0.0] |[0.9998544978598302,1.4550214016984225E-4,0.0]|0.0       |
|110414.71|CASH_OUT|26845.41     |0.00          |288800.00     |2415.16       |0.0       |[110414.71,26845.41,0.0,288800.0,2415.16,0.0]|[19.978956525017455,0.02104347498254672,0.0]  |[0.9989478262508726,0.001052173749127336,0.0] |0.0       |
|56953.90 |CASH_OUT|1942.02      |0.00          |70253.00      |64106.18      |0.0       |[56953.9,1942.02,0.0,70253.0,64106.18,0.0]   |[19.984101757507357,0.015898242492648023,0.0] |[0.9992050878753677,7.94912124632401E-4,0.0]  |0.0       |
|5346.89  |CASH_OUT|0.00         |0.00          |652637.00     |6453430.91    |0.0       |[5346.89,0.0,0.0,652637.0,6453430.91,0.0]    |[19.98976921686754,0.010230783132462807,0.0]  |[0.9994884608433768,5.115391566231402E-4,0.0] |0.0       |
|23261.30 |CASH_OUT|20411.53     |0.00          |25742.00      |0.00          |0.0       |[23261.3,20411.53,0.0,25742.0,0.0,0.0]       |[19.969589817966106,0.030410182033893656,0.0] |[0.9984794908983053,0.0015205091016946828,0.0]|0.0       |
|62610.80 |TRANSFER|79114.00     |16503.20      |517.00        |8383.29       |1.0       |[62610.8,79114.0,16503.2,517.0,8383.29,1.0]  |[19.996094960417096,0.0039050395829045327,0.0]|[0.9998047480208548,1.9525197914522664E-4,0.0]|0.0       |
+---------+--------+-------------+--------------+--------------+--------------+----------+---------------------------------------------+----------------------------------------------+----------------------------------------------+----------+


     */
    df3.createOrReplaceTempView("df3")
    val df4 = spark.sql("select amount,type,type_label,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,prediction from df3")
    println("new_data_set_final")
    df4.show(10,false)
    /*
+---------+--------+----------+-------------+--------------+--------------+--------------+----------+
|amount   |type    |type_label|oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|prediction|
+---------+--------+----------+-------------+--------------+--------------+--------------+----------+
|181.00   |TRANSFER|1.0       |181.00       |0.00          |0.00          |0.00          |0.0       |
|181.00   |CASH_OUT|0.0       |181.00       |0.00          |21182.00      |0.00          |0.0       |
|229133.94|CASH_OUT|0.0       |15325.00     |0.00          |5083.00       |51513.44      |0.0       |
|215310.30|TRANSFER|1.0       |705.00       |0.00          |22425.00      |0.00          |0.0       |
|311685.89|TRANSFER|1.0       |10835.00     |0.00          |6267.00       |2719172.89    |0.0       |
|110414.71|CASH_OUT|0.0       |26845.41     |0.00          |288800.00     |2415.16       |0.0       |
|56953.90 |CASH_OUT|0.0       |1942.02      |0.00          |70253.00      |64106.18      |0.0       |
|5346.89  |CASH_OUT|0.0       |0.00         |0.00          |652637.00     |6453430.91    |0.0       |
|23261.30 |CASH_OUT|0.0       |20411.53     |0.00          |25742.00      |0.00          |0.0       |
|62610.80 |TRANSFER|1.0       |79114.00     |16503.20      |517.00        |8383.29       |0.0       |
+---------+--------+----------+-------------+--------------+--------------+--------------+----------+
     */
    df4.createOrReplaceTempView("new_dataset_view")
    println("df4 describe")
    df4.describe().show()
    /*
   +-------+-----------------+--------+-------------------+------------------+-----------------+-----------------+-----------------+--------------------+
|summary|           amount|    type|         type_label|     oldbalanceOrg|   newbalanceOrig|   oldbalanceDest|   newbalanceDest|          prediction|
+-------+-----------------+--------+-------------------+------------------+-----------------+-----------------+-----------------+--------------------+
|  count|          2770409| 2770409|            2770409|           2770409|          2770409|          2770409|          2770409|             2770409|
|   mean|    317536.140869|    null|0.19235751833032597|      47643.079411|     16091.904679|   1703551.161910|   2049734.436887|0.001326879893907...|
| stddev|887789.6576433469|    null| 0.3941524572409774|251325.12749238333|151255.8247602122|4225550.490196659|4676990.061142499| 0.03640219446635694|
|    min|             0.00|CASH_OUT|                0.0|              0.00|             0.00|             0.00|             0.00|                 0.0|
|    max|      92445516.64|TRANSFER|                1.0|       59585040.37|      49585040.37|     356015889.35|     356179278.92|                 1.0|
+-------+-----------------+--------+-------------------+------------------+-----------------+-----------------+-----------------+--------------------+
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
|2.15  |CASH_OUT|22958.00     |22955.85      |0.00          |2.15          |0.0       |0      |
|6.83  |CASH_OUT|0.00         |0.00          |151002.00     |13941.51      |0.0       |0      |
|7.23  |TRANSFER|0.00         |0.00          |426840.94     |426848.17     |0.0       |0      |
|8.67  |CASH_OUT|0.00         |0.00          |5730383.88    |5730392.55    |0.0       |0      |
|12.10 |CASH_OUT|0.00         |0.00          |51327.08      |51339.18      |0.0       |0      |
|23.93 |CASH_OUT|211752.85    |211728.92     |163404.55     |163428.47     |0.0       |0      |
|23.99 |CASH_OUT|0.00         |0.00          |91775.91      |91799.90      |0.0       |0      |
|26.23 |CASH_OUT|10505.00     |10478.77      |0.00          |26.23         |0.0       |0      |
|27.17 |CASH_OUT|0.00         |0.00          |1982814.62    |1982841.79    |0.0       |0      |
|30.27 |TRANSFER|0.00         |0.00          |18997.38      |19027.65      |0.0       |0      |
|31.73 |CASH_OUT|0.00         |0.00          |101066.80     |101098.53     |0.0       |0      |
|31.95 |CASH_OUT|0.00         |0.00          |539253.56     |539285.51     |0.0       |0      |
|35.53 |CASH_OUT|42255.00     |42219.47      |819429.81     |819465.34     |0.0       |0      |
|35.94 |CASH_OUT|0.00         |0.00          |374457.73     |374493.67     |0.0       |0      |
|36.05 |CASH_OUT|0.00         |0.00          |597098.33     |597134.38     |0.0       |0      |
|38.51 |CASH_OUT|0.00         |0.00          |17104.19      |17142.71      |0.0       |0      |
|39.43 |CASH_OUT|11354.00     |11314.57      |4268907.60    |4140999.89    |0.0       |0      |
|40.60 |CASH_OUT|209899.00    |209858.40     |30.88         |71.48         |0.0       |0      |
|44.31 |CASH_OUT|14506.00     |14461.69      |0.00          |44.31         |0.0       |0      |
|44.59 |CASH_OUT|0.00         |0.00          |147621.69     |147666.27     |0.0       |0      |
|45.42 |CASH_OUT|19180.00     |19134.58      |534698.18     |534743.60     |0.0       |0      |
|46.41 |CASH_OUT|0.00         |0.00          |339113.74     |339160.15     |0.0       |0      |
|46.66 |CASH_OUT|12748.00     |12701.34      |282786.87     |282833.53     |0.0       |0      |
|47.03 |CASH_OUT|0.00         |0.00          |18318.42      |18365.45      |0.0       |0      |
|48.01 |CASH_OUT|0.00         |0.00          |6207921.63    |6207969.63    |0.0       |0      |
+------+--------+-------------+--------------+--------------+--------------+----------+-------+
     */
  println("prediction filter")
    final_out.filter( "prediction = 1").show(50,false)
    println("filter isfraud")
    /*
    +----------+--------+-------------+--------------+--------------+--------------+----------+-------+
|amount    |type    |oldbalanceOrg|newbalanceOrig|oldbalanceDest|newbalanceDest|prediction|isFraud|
+----------+--------+-------------+--------------+--------------+--------------+----------+-------+
|1335602.19|TRANSFER|796471.56    |0.00          |0.00          |1293572.49    |1.0       |0      |
|963532.14 |CASH_OUT|963532.14    |0.00          |132382.57     |1095914.71    |1.0       |1      |
|1324918.08|TRANSFER|811465.65    |0.00          |3754771.25    |5369750.26    |1.0       |0      |
|1019603.21|TRANSFER|764178.26    |0.00          |0.00          |1019603.21    |1.0       |0      |
|1265279.12|TRANSFER|817680.66    |0.00          |16861402.92   |19113578.52   |1.0       |0      |
|1657791.60|TRANSFER|995665.65    |0.00          |0.00          |1703266.91    |1.0       |0      |
|1027908.63|TRANSFER|970574.95    |0.00          |106445.00     |1134353.63    |1.0       |0      |
|1300410.07|TRANSFER|1223229.87   |0.00          |3479380.26    |5205572.90    |1.0       |0      |
|2154039.07|TRANSFER|868850.34    |0.00          |6038983.64    |9128938.99    |1.0       |0      |
|2284539.84|TRANSFER|1023860.79   |0.00          |2641758.09    |8953314.93    |1.0       |0      |
|1743026.49|TRANSFER|774992.82    |0.00          |0.00          |1743026.49    |1.0       |0      |
|1954359.42|TRANSFER|1836964.18   |0.00          |2526131.00    |4480490.42    |1.0       |0      |
|1465010.08|TRANSFER|1264518.58   |0.00          |1582072.83    |4185266.80    |1.0       |0      |
|951086.80 |TRANSFER|801562.26    |0.00          |1447340.41    |2753755.78    |1.0       |0      |
|2930418.44|CASH_OUT|2930418.44   |0.00          |3616012.10    |6963508.84    |1.0       |1      |
|2035123.69|TRANSFER|1753091.06   |0.00          |6314940.34    |8090741.06    |1.0       |0      |
|2068118.36|TRANSFER|2068118.36   |0.00          |0.00          |0.00          |1.0       |1      |
|2627070.50|TRANSFER|2627070.50   |0.00          |0.00          |0.00          |1.0       |1      |
|938288.58 |TRANSFER|938288.58    |0.00          |0.00          |0.00          |1.0       |1      |
|1014816.16|TRANSFER|1014816.16   |0.00          |0.00          |0.00          |1.0       |1      |
|2023920.09|TRANSFER|2023920.09   |0.00          |0.00          |0.00          |1.0       |1      |
|1789155.73|TRANSFER|1789155.73   |0.00          |0.00          |0.00          |1.0       |1      |
|2694857.26|TRANSFER|2694857.26   |0.00          |0.00          |0.00          |1.0       |1      |
|1218165.88|CASH_OUT|1218165.88   |0.00          |742564.27     |1960730.15    |1.0       |1      |
|810039.19 |TRANSFER|810039.19    |0.00          |0.00          |0.00          |1.0       |1      |
|1461930.03|CASH_OUT|1461930.03   |0.00          |3762496.73    |5224426.76    |1.0       |1      |
|1844357.26|CASH_OUT|1844357.26   |0.00          |3549005.46    |5393362.71    |1.0       |1      |
|3857176.83|CASH_OUT|3857176.83   |0.00          |228541.62     |4085718.45    |1.0       |1      |
|1170282.92|CASH_OUT|1170282.92   |0.00          |173264.14     |1343547.06    |1.0       |1      |
|3760068.04|CASH_OUT|3760068.04   |0.00          |622555.65     |4382623.69    |1.0       |1      |
|1076739.91|TRANSFER|1076739.91   |0.00          |0.00          |0.00          |1.0       |1      |
|960306.85 |CASH_OUT|960306.85    |0.00          |0.00          |960306.85     |1.0       |1      |
|865948.49 |TRANSFER|865948.49    |0.00          |0.00          |0.00          |1.0       |1      |
|1103343.59|TRANSFER|814907.93    |0.00          |3663110.79    |4766454.38    |1.0       |0      |
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

    val df3_man = model.transform(df2_man)
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
