from pyspark.sql import SparkSession
from src.data.dataset_builder import process_location_and_build_dataset

def main():
    spark = SparkSession.builder.appName("OilSpillDatasetCollector").getOrCreate()
    sc = spark.sparkContext

    locations = [
        ("Gulf_of_Mexico", [-90.0, 28.0, -89.9, 28.1]),
        ("Port_of_LA", [-118.3, 33.7, -118.2, 33.8]),
        ("Mumbai_Coast", [72.8, 18.9, 72.9, 19.0]),
        ("Niger_Delta", [6.0, 4.8, 6.1, 4.9]),
        ("Persian_Gulf", [50.0, 26.5, 50.1, 26.6]),
        ("Singapore_Strait", [103.7, 1.2, 103.8, 1.3]),
        ("Rotterdam_Port", [4.3, 51.9, 4.4, 52.0]),
        ("Alaska_Coast", [-150.0, 60.9, -149.9, 61.0])
    ]
    
    locations_rdd = sc.parallelize(locations)
    
    # Запускаем параллельную обработку
    results = locations_rdd.map(process_location_and_build_dataset).collect()
    
    for res in results:
        print(res)

if __name__ == "__main__":
    main()