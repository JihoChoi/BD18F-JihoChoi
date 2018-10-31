#untar
#tar -zcvf nemo-shaded.tar.gz ./nemo-shaded.jar
tar -zxvf nemo-shaded.tar.gz

#install
mvn clean install -Pnemo-runner

#run

: '
# Original
java -cp `pwd`/nemo-shaded.jar:`yarn classpath`:`pwd`/target/bd_hw-1.0-SNAPSHOT.jar \
org.apache.nemo.client.JobLauncher \
  -job_id BD_HW_ONE \
  -executor_json `pwd`/nemo-resources.json \
  -optimization_policy org.apache.nemo.compiler.optimizer.policy.DefaultPolicy \
  -user_main edu.snu.bd.examples.Homework \
  -user_args "--runner=org.apache.nemo.compiler.frontend.beam.NemoPipelineRunner --inputFile=`pwd`/input.txt --output=`pwd`/nemo_output/output"
'

: '
java -cp `pwd`/nemo-shaded.jar:`yarn classpath`:`pwd`/target/bd_hw-1.0-SNAPSHOT.jar \
org.apache.nemo.client.JobLauncher \
  -job_id BD_HW_ONE \
  -executor_json `pwd`/nemo-resources.json \
  -optimization_policy org.apache.nemo.compiler.optimizer.policy.DefaultPolicy \
  -user_main edu.snu.bd.hw1.Main \
  -user_args "--runner=org.apache.nemo.compiler.frontend.beam.NemoPipelineRunner --inputFile=`pwd`/input.txt --output=`pwd`/nemo_output/output"
'

# java -cp `pwd`/nemo-shaded.jar:`yarn classpath`:`pwd`/target/bd_hw-1.0-SNAPSHOT.jar \

java -cp `pwd`/nemo-shaded.jar:`yarn classpath`:`pwd`/target/bd_hw-1.0-SNAPSHOT-shaded.jar \
org.apache.nemo.client.JobLauncher \
  -job_id BD_HW_ONE \
  -executor_json `pwd`/nemo-resources.json \
  -optimization_policy org.apache.nemo.compiler.optimizer.policy.DefaultPolicy \
  -user_main edu.snu.bd.hw1.Main \
  -user_args "--runner=org.apache.nemo.compiler.frontend.beam.NemoPipelineRunner --inputFile_ranking=`pwd`/data/fifa_ranking.csv --inputFile_players=`pwd`/data/wc2018_players.csv --output=`pwd`/nemo_output/output"

