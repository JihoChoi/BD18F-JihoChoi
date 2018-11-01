/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.snu.bd.hw1;

import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.PipelineResult;
import org.apache.beam.sdk.extensions.sql.SqlTransform;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.metrics.Counter;
import org.apache.beam.sdk.metrics.Distribution;
import org.apache.beam.sdk.metrics.Metrics;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.options.Validation.Required;
import org.apache.beam.sdk.schemas.Schema;
import org.apache.beam.sdk.transforms.*;
import org.apache.beam.sdk.values.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.FileNotFoundException;

import java.util.*;

import org.apache.commons.csv.CSVFormat;
import org.apache.beam.sdk.extensions.sql.meta.provider.text.TextTableProvider;
import org.apache.beam.sdk.coders.*;
import org.apache.beam.sdk.transforms.*;
import org.apache.beam.sdk.coders.*;
import org.apache.beam.sdk.schemas.*;
import org.apache.beam.sdk.values.*;


/**
 * An example that counts words in Shakespeare and includes Beam best practices.
 *
 * <p>This class, {@link Main}, is the second in a series of four successively more detailed
 * 'word count' examples. You may first want to take a look at MinimalWordCount. After
 * you've looked at this example, then see the DebuggingWordCount pipeline, for introduction
 * of additional concepts.
 *
 * <p>For a detailed walkthrough of this example, see <a
 * href="https://beam.apache.org/get-started/wordcount-example/">
 * https://beam.apache.org/get-started/wordcount-example/ </a>
 *
 * <p>Basic concepts, also in the MinimalWordCount example: Reading text files; counting a
 * PCollection; writing to text files
 *
 * <p>New Concepts:
 *
 * <pre>
 *   1. Executing a Pipeline both locally and using the selected runner
 *   2. Using ParDo with static DoFns defined out-of-line
 *   3. Building a composite transform
 *   4. Defining your own pipeline options
 * </pre>
 *
 * <p>Concept #1: you can execute this pipeline either locally or using by selecting another runner.
 * These are now command-line options and not hard-coded as they were in the MinimalWordCount
 * example.
 *
 * <p>To change the runner, specify:
 *
 * <pre>{@code
 * --runner=YOUR_SELECTED_RUNNER
 * }</pre>
 *
 * <p>To execute this pipeline, specify a local output file (if using the {@code DirectRunner}) or
 * output prefix on a supported distributed file system.
 *
 * <pre>{@code
 * --output=[YOUR_LOCAL_FILE | YOUR_OUTPUT_PREFIX]
 * }</pre>
 *
 * <p>The input file defaults to a public data set containing the text of of King Lear, by William
 * Shakespeare. You can override it and choose your own input with {@code --inputFile}.
 */
public class Main {

  public static final String TOKENIZER_PATTERN = "[^\\p{L}]+";

  /**
   * Concept #2: You can make your pipeline assembly code less verbose by defining your DoFns
   * statically out-of-line. This DoFn tokenizes lines of text into individual words; we pass it to
   * a ParDo in the pipeline.
   */
  static class ExtractWordsFn extends DoFn<String, String> {
    private final Counter emptyLines = Metrics.counter(ExtractWordsFn.class, "emptyLines");
    private final Distribution lineLenDist =
            Metrics.distribution(ExtractWordsFn.class, "lineLenDistro");

    @ProcessElement
    public void processElement(@Element String element, OutputReceiver<String> receiver) {
      lineLenDist.update(element.length());
      if (element.trim().isEmpty()) {
        emptyLines.inc();
      }

      // Split the line into words.
      String[] words = element.split(TOKENIZER_PATTERN, -1);

      // Output each word encountered into the output PCollection.
      for (String word : words) {
        if (!word.isEmpty()) {
          receiver.output(word);
        }
      }
    }
  }

  /** A SimpleFunction that converts a Word and Count into a printable string. */
  public static class FormatAsTextFn extends SimpleFunction<KV<String, Long>, String> {
    @Override
    public String apply(KV<String, Long> input) {
      return input.getKey() + ": " + input.getValue();
    }
  }

  /**
   * A PTransform that converts a PCollection containing lines of text into a PCollection of
   * formatted word counts.
   *
   * <p>Concept #3: This is a custom composite transform that bundles two transforms (ParDo and
   * Count) as a reusable PTransform subclass. Using composite transforms allows for easy reuse,
   * modular testing, and an improved monitoring experience.
   */
  public static class CountWords extends PTransform<PCollection<String>, PCollection<KV<String, Long>>> {
    @Override
    public PCollection<KV<String, Long>> expand(PCollection<String> lines) {

      // Convert lines of text into individual words.
      PCollection<String> words = lines.apply(ParDo.of(new ExtractWordsFn()));

      // Count the number of times each word occurs.
      PCollection<KV<String, Long>> wordCounts = words.apply(Count.perElement());

      return wordCounts;
    }
  }

/*
  public interface PipelineOptions extends org.apache.beam.sdk.options.PipelineOptions {
    @Description("Path of the file to read from")
    @Required
    String getInputFile();

    void setInputFile(String value);

    @Description("Path of the file to write to")
    @Required
    String getOutput();

    void setOutput(String value);
  }
*/


  public interface PipelineOptions extends org.apache.beam.sdk.options.PipelineOptions {

    @Description("Path of the ranking file to read from")
    @Required
    String getInputFile_ranking();

    void setInputFile_ranking(String value);

    @Description("Path of the players file to read from")
    @Required
    String getInputFile_players();

    void setInputFile_players(String value);

    @Description("Path of the file to write to")
    @Required
    String getOutput();

    void setOutput(String value);
  }

  /*
  static void applyWordCountLogic(final Pipeline pipeline, final String input, final String output) {
    pipeline.apply("ReadLines", TextIO.read().from(input))
            .apply(new CountWords())
            .apply(MapElements.via(new FormatAsTextFn()))
            .apply("WriteCounts", TextIO.write().to(output));
  }
  */

  static void applyWorldCupCalcLogic(final Pipeline pipeline,
                                     final String input_ranking,
                                     final String input_players,
                                     final String output) {

    System.out.println("DEBUG > apply > begin ================================================================= JIHO");
    System.out.println("1) " + input_ranking + "\n2) " + input_players);

    // CountWords
    // pipeline.apply("ReadLines", TextIO.read().from(input_ranking))
    //         .apply(new CountWords())
    //         .apply(MapElements.via(new FormatAsTextFn()))
    //         .apply("WriteCounts", TextIO.write().to(output));

    // Load only the required fields from .csv files and create PCollection<Row>

    /*
    References
      https://beam.apache.org/documentation/dsls/sql/walkthrough/
    */

    /*
    // Undistributed in-memory load

    List<Row> rows = new ArrayList<Row>();
    String line = "";
    try (BufferedReader br = new BufferedReader(new FileReader(input_ranking))) {
      line = br.readLine(); // Remove Header
      while ((line = br.readLine()) != null) {
        String[] entries = line.split(",");
        // System.out.println(entries[0] + " " + entries[1] + " " + entries[15]);
        // row = Row.withSchema(rankingSchema).addValues(entries[0], entries[1], entries[15]).build();
        rows.add(Row.withSchema(rankingSchema).addValues(Integer.parseInt(entries[0]), entries[1], entries[15]).build());
      }
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      System.out.println("##### Can Not Load Data #####");
    }

    PCollection<Row> table = GenericSourceSink.read(pipeline, filePattern)
            .apply("StringToRow", new TextTableProvider.CsvToRow(tableSchema.getValue(), csvFormat))
            .setCoder(tableSchema.getValue().getRowCoder())
            .setName(tableSchema.getKey());
    */

    // Load CSV
    //    Refernce: https://github.com/apache/incubator-nemo/blob/78e182c98554e4cdea6e63b0a9fed4905e75e2f2/examples/beam/src/main/java/org/apache/nemo/examples/beam/tpch/TpchQueryRunner.java#L75-L78

    // rank,country_full,country_abrv,total_points,previous_points,rank_change,cur_year_avg,cur_year_avg_weighted,last_year_avg,last_year_avg_weighted,two_year_ago_avg,two_year_ago_weighted,three_year_ago_avg,three_year_ago_weighted,confederation,rank_date
    Schema RANKING_SCHEMA = Schema.builder()
            .addInt32Field("rank_num")     // 0
            .addStringField("country") // 1
            .addStringField("temp2").addStringField("temp3").addStringField("temp4")
            .addStringField("temp5").addStringField("temp6").addStringField("temp7")
            .addStringField("temp8").addStringField("temp9").addStringField("temp10")
            .addStringField("temp11").addStringField("temp12").addStringField("temp13").addStringField("temp14")
            .addStringField("rank_date")    // 15
            .build();

    // Team,#,Pos.,FIFA Popular Name,Birth Date,Shirt Name,Club,Height,Weight
    Schema PLAYER_SCHEMA = Schema.builder()
            .addStringField("country") // 0
            .addStringField("temp1").addStringField("temp2").addStringField("temp3")
            .addStringField("temp4").addStringField("temp5").addStringField("temp6")
            .addDoubleField("height") // 7
            .addDoubleField("weight") // 8
            .build();

    Schema PLAYER_SCHEMA_2 = Schema.builder()
            .addStringField("country")
            .addDoubleField("height")
            .addDoubleField("weight")
            .build();

    final CSVFormat csvFormat = CSVFormat.MYSQL
            .withDelimiter(',')
            .withNullString("")
            .withTrailingDelimiter();

    PCollectionTuple tables = PCollectionTuple.empty(pipeline);

    final PCollection<Row> table_ranking = pipeline.apply(TextIO.read().from(input_ranking))
            .apply("StringToRow", new TextTableProvider.CsvToRow(RANKING_SCHEMA, csvFormat))
            .setCoder(RANKING_SCHEMA.getRowCoder())
            .setName("ranking");

    tables = tables.and(new TupleTag<>("ranking"), table_ranking);

    final PCollection<Row> table_player = pipeline.apply(TextIO.read().from(input_players))
            .apply("StringToRow", new TextTableProvider.CsvToRow(PLAYER_SCHEMA, csvFormat))
            .setCoder(PLAYER_SCHEMA.getRowCoder())
            .setName("player");

    tables = tables.and(new TupleTag<>("player"), table_player);

    // Query 1 Ranking
    PCollection<Row> outputStream_ranking = tables.apply(
            SqlTransform.query("select rank_num, country from ranking where rank_date = '2018-06-07'"));

    // Query 2 Player
    PCollection<Row> outputStream_player = tables
            .apply(SqlTransform.query("select country, avg(height), avg(weight) from player group by country"))
            .setCoder(PLAYER_SCHEMA_2.getRowCoder());

    // Join Query
    PCollectionTuple tables_results = PCollectionTuple.empty(pipeline);
    tables_results = tables_results.and(new TupleTag<>("ranking"), outputStream_ranking);
    tables_results = tables_results.and(new TupleTag<>("player"), outputStream_player);

    PCollection<Row> outputStream = tables_results.apply(
            SqlTransform.query("select rank_num, ranking.country, height, player.weight " +
                    "from ranking " +
                    "inner join player " +
                    "on ranking.country = player.country"));

    // outputStream_player.apply(
    // outputStream_ranking.apply(

    outputStream.apply(
            "log_result",
            MapElements.via(
                    new SimpleFunction<Row, Void>() {
                      @Override
                      public Void apply(Row input) {
                        System.out.println("+----------------------------------------------------");
                        System.out.println("| PCOLLECTION: " + input.getValues());
                        System.out.println("+----------------------------------------------------");
                        return null;
                      }
                    }));

    // new TupleTag<>("ranking"), table_ranking



    System.out.println("DEBUG > apply > end =================================================================== JIHO");
  }

  /**
   * This is a quick example, which uses Beam SQL DSL to create a data pipeline.
   *
   * <p>Run the example from the Beam source root with
   *
   * <pre>
   *   ./gradlew :beam-sdks-java-extensions-sql:runBasicExample
   * </pre>
   *
   * <p>The above command executes the example locally using direct runner. Running the pipeline in
   * other runners require additional setup and are out of scope of the SQL examples. Please consult
   * Beam documentation on how to run pipelines.
   */
  static void applySQLLogic(final Pipeline p) {
    //define the input row format
    Schema type =
            Schema.builder().addInt32Field("c1").addStringField("c2").addDoubleField("c3").build();

    Row row1 = Row.withSchema(type).addValues(1, "row", 1.0).build();
    Row row2 = Row.withSchema(type).addValues(2, "row", 2.0).build();
    Row row3 = Row.withSchema(type).addValues(3, "row", 3.0).build();

    //create a source PCollection with Create.of();
    PCollection<Row> inputTable =
            PBegin.in(p)
                    .apply(
                            Create.of(row1, row2, row3)
                                    .withSchema(
                                            type, SerializableFunctions.identity(), SerializableFunctions.identity()))
                    .setCoder(type.getRowCoder());

    //Case 1. run a simple SQL query over input PCollection with BeamSql.simpleQuery;
    PCollection<Row> outputStream =
            inputTable.apply(SqlTransform.query("select c1, c2, c3 from PCOLLECTION where c1 > 1"));

    // print the output record of case 1;
    outputStream.apply(
            "log_result",
            MapElements.via(
                    new SimpleFunction<Row, Void>() {
                      @Override
                      public Void apply(Row input) {
                        // expect output:
                        //  PCOLLECTION: [3, row, 3.0]
                        //  PCOLLECTION: [2, row, 2.0]

                        System.out.println("");
                        System.out.println("+----------------------------------------------------");
                        System.out.println("| PCOLLECTION: " + input.getValues());
                        System.out.println("+----------------------------------------------------");

                        return null;
                      }
                    }));

    // Case 2. run the query with SqlTransform.query over result PCollection of case 1.
    PCollection<Row> outputStream2 =
            PCollectionTuple.of(new TupleTag<>("CASE1_RESULT"), outputStream)
                    .apply(SqlTransform.query("select c2, sum(c3) from CASE1_RESULT group by c2"));

    // print the output record of case 2;
    outputStream2.apply(
            "log_result",
            MapElements.via(
                    new SimpleFunction<Row, Void>() {
                      @Override
                      public Void apply(Row input) {
                        // expect output:
                        //  CASE1_RESULT: [row, 5.0]
                        System.out.println("+-------------------------------");
                        System.out.println("CASE1_RESULT: " + input.getValues());
                        System.out.println("+-------------------------------");
                        return null;
                      }
                    }));
  }


  //  static void runWordCount(String[] args) throws Exception {
  //    PipelineOptions options =
  //            PipelineOptionsFactory.fromArgs(args).withValidation().as(PipelineOptions.class);
  //    Pipeline p = Pipeline.create(options);
  //
  //    applyWordCountLogic(p, options.getInputFile(), options.getOutput());
  //
  //    final PipelineResult result =  p.run();
  //    if (args[0].equals("--runner=SparkRunner")) {
  //      result.waitUntilFinish(); // Not yet supported by Nemo.
  //    }
  //  }
  //
  static void runSQL(String[] args) throws Exception {
    PipelineOptions options =
            PipelineOptionsFactory.fromArgs(args).withValidation().as(PipelineOptions.class);
    Pipeline p = Pipeline.create(options);
    applySQLLogic(p);
    final PipelineResult result =  p.run();
    if (args[0].equals("--runner=SparkRunner")) {
      result.waitUntilFinish();
    }
  }


  static void runWorldCupCalc(String[] args) throws Exception {
    //    PipelineOptions options = PipelineOptionsFactory.fromArgs(args).withValidation().as(PipelineOptions.class);

    PipelineOptions options =
            PipelineOptionsFactory.fromArgs(args).withValidation().as(PipelineOptions.class);
    Pipeline p = Pipeline.create(options);

    System.out.println("DEBUG > =================================================================== JIHO");

    applyWorldCupCalcLogic(p, options.getInputFile_ranking(), options.getInputFile_players(), options.getOutput());

    System.out.println("DEBUG > =================================================================== JIHO");

    final PipelineResult result =  p.run();

    // System.out.println("PIPELINE RESULT: " + result);

    System.out.println("DEBUG > =================================================================== JIHO");

    if (args[0].equals("--runner=SparkRunner")) {
      result.waitUntilFinish(); // Not yet supported by Nemo.
    }
  }


  /**
   * Big Data and Deep Learning Systems Fall 2018
   * ================= Write your data processing logic here =================
   */
  public static void main(String[] args) throws Exception {

    System.out.println("+------------------------------------------------------");
    System.out.println("| HW01 -  Beam on Spark / Nemo");
    System.out.println("|    Jiho Choi (jihochoi@snu.ac.kr)");
    System.out.println("|");
    System.out.println("+------------------------------------------------------");
    System.out.println("|    START");
    System.out.println("+------------------------------------------------------");
    System.out.println("");

    System.out.println(args[0]); // spark
    System.out.println(args[1]); // path_rank_data
    System.out.println(args[2]); // path_player_data
    System.out.println(args[3] + "\n"); // path_output_dir

    runWorldCupCalc(args);

    // runWordCount(args);
    // runSQL(args);

    System.out.println("");
    System.out.println("+------------------------------------------------------");
    System.out.println("|    END");
    System.out.println("+------------------------------------------------------");

  }
}

