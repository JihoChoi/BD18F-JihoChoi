# Install and Run

# mvn compile exec:java -Dexec.mainClass=edu.snu.bd.examples.Homework \
#    -Dexec.args="--runner=SparkRunner --inputFile=input.txt --output=`pwd`/spark_output/output" \
#    -Pspark-runner



# mvn compile exec:java -Dexec.mainClass=edu.snu.bd.hw1.Main \
#     -Dexec.args="--runner=SparkRunner --inputFile=input.txt --output=`pwd`/spark_output/output" \
#     -Pspark-runner



# -Dexec.args="--runner=SparkRunner --inputFile_ranking=./data/fifa_ranking.csv --inputFile_players=./data/wc2018_players.csv --output=`pwd`/spark_output/output" \
mvn compile exec:java -Dexec.mainClass=edu.snu.bd.hw1.Main \
    -Dexec.args="--runner=SparkRunner --inputFile_ranking=./data/ranking.csv --inputFile_players=./data/players.csv --output=`pwd`/spark_output/output" \
    -Pspark-runner
