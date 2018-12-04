# BD18F-JihoChoi

### Programming Assignments Repository

Topics in Big Data Analytics (Big Data Analytics and Deep Learning Systems)

BDDL Fall 2018 @SNU
Course GitHub: https://github.com/swsnu/bd2018



## Course

* Resource management
    * YARN, Mesos, Borg
* Meta-framework
    * REEF
* Dataflow Processing Framework
    * MapReduce, Dryad, Spark
* High-level Data Processing
    * Hive, Pig, FlumeJava, DryadLINQ, Beam
* Stream Processing
    * Strom, Heron, SparkStreaming, Flink, MillWheel, Dataflow, Samza
* Machine Learning / Deep Learning Systems
    * . TBA


## [HW1] Beam on Spark / Nemo

### Description
Simple BeamSQL batch processing application for distributed query processing using Beam / Spark / Apache Nemo

### Requirements
- JRE 1.8
- Maven
    - ND4J, DL4J

### Usage
```
> ./download_datasets.sh
> ./run_spark
> ./run_nemo
```

### Datasets
* `fifa-ranking.csv`
  * FIFA Rankings by Rank Dates (1993-2018)
  * [Kaggle FIFA Ranking](https://www.kaggle.com/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now)
* `wc2018-players.csv`
  * 2018 FIFA World Cup Players (32 countries * 23-man)
  * [Kaggle FIFA WORLD CUP 2018 Players](https://www.kaggle.com/djamshed/fifa-world-cup-2018-players)

#### Querying without parallel processing
> . simple queries  
> SELECT rank_num, country FROM RANKING WHEN rank_date = '2018-06-07'  
> SELECT rank_num, RANKING.COUNTRY, PLAYER.height, PLAYER.weight FROM PLAYER GROUP BY country"  
> SELECT rank_num, RANKING.COUNTRY, PLAYER.height, PLAYER.weight, BMI(PLAYER.height, PLAYER.weight) FROM RANKING INNER JOIN PLAYER ON RANKING.country = PLAYER.country  

#### Data flow with distributed systems
> . requires a lot more  
> PIPELINE_OPTION  
> PIPELINE  
> PCOLLECTION  
> PTRANSFORMATION  


.
---

## [HW2] Deep Learning
