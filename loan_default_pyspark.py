from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt

# Start Spark session
spark = SparkSession.builder.appName("Loan_Default_Predictor").getOrCreate()

# Load dataset (dataset should be in same folder)
data_path = "Loan_default.csv"
data = spark.read.csv(data_path, header=True, inferSchema=True)

# Convert target column to numeric
data = data.withColumn("Default", col("Default").cast("double"))

# Handle missing categorical values
cat_columns = [
    "Education", "EmploymentType", "MaritalStatus",
    "LoanPurpose", "HasMortgage", "HasDependents", "HasCoSigner"
]
data = data.fillna({c: "Unknown" for c in cat_columns})

# Handle missing numeric values using median
num_columns = ["LoanAmount", "CreditScore", "Income", "DTIRatio"]
for feature in num_columns:
    median_val = data.approxQuantile(feature, [0.5], 0.25)[0]
    data = data.withColumn(
        feature,
        when(isnan(col(feature)) | col(feature).isNull(), median_val)
        .otherwise(col(feature))
    )

# Convert Yes/No columns to numeric
data = data.withColumn(
    "HasMortgage_val",
    when(col("HasMortgage").isin("Yes", "Y", "yes"), 1.0).otherwise(0.0)
)

data = data.withColumn(
    "HasDependents_val",
    when(col("HasDependents").isin("Yes", "Y", "yes"), 1.0).otherwise(0.0)
)

data = data.withColumn(
    "HasCoSigner_val",
    when(col("HasCoSigner").isin("Yes", "Y", "yes"), 1.0).otherwise(0.0)
)

# Encode categorical features
edu_indexer = StringIndexer(
    inputCol="Education", outputCol="Education_Idx", handleInvalid="keep"
)
emp_indexer = StringIndexer(
    inputCol="EmploymentType", outputCol="Employment_Idx", handleInvalid="keep"
)
mar_indexer = StringIndexer(
    inputCol="MaritalStatus", outputCol="Marital_Idx", handleInvalid="keep"
)
purpose_indexer = StringIndexer(
    inputCol="LoanPurpose", outputCol="Purpose_Idx", handleInvalid="keep"
)

# Remove outliers from LoanAmount using IQR
q1, q3 = data.approxQuantile("LoanAmount", [0.25, 0.75], 0.0)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
data = data.filter(
    (col("LoanAmount") >= lower) & (col("LoanAmount") <= upper)
)

# Feature list
input_features = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
    "Education_Idx", "Employment_Idx", "Marital_Idx",
    "HasMortgage_val", "HasDependents_val", "HasCoSigner_val", "Purpose_Idx"
]

# Assemble features
assembler = VectorAssembler(
    inputCols=input_features,
    outputCol="features",
    handleInvalid="keep"
)

# Model
log_reg = LogisticRegression(
    featuresCol="features",
    labelCol="Default",
    probabilityCol="probability"
)

# Pipeline
pipeline = Pipeline(stages=[
    edu_indexer,
    emp_indexer,
    mar_indexer,
    purpose_indexer,
    assembler,
    log_reg
])

# Train-test split
train_set, test_set = data.randomSplit([0.7, 0.3], seed=42)

# Train model
trained_model = pipeline.fit(train_set)

# Predictions
preds = trained_model.transform(test_set)
preds.select("Default", "prediction", "probability").show(10, truncate=False)

# Evaluate model
evaluator = BinaryClassificationEvaluator(
    labelCol="Default",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc_score = evaluator.evaluate(preds)
print(f"\nModel AUC Score: {auc_score:.3f}\n")

# Plot actual vs predicted default ratios
actual_ratio = test_set.filter(col("Default") == 1).count() / test_set.count()
pred_ratio = preds.filter(col("prediction") == 1).count() / preds.count()

plt.bar(
    ['Actual Defaults', 'Predicted Defaults'],
    [actual_ratio, pred_ratio]
)
plt.ylabel("Default Ratio")
plt.title("Actual vs Predicted Default Comparison")
plt.show()

# Stop Spark session
spark.stop()
