import java.io.IOException;
import java.util.StringTokenizer;

import javax.naming.Context;

//import javax.naming.Context;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class WordCount {

  public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {

    private final static IntWritable one = new IntWritable(1);
    private final Text word = new Text();

    @Override
    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {

        // Convert Text to String
        String line = value.toString();

        // Remove punctuation using regex
        line = line.replaceAll("[^a-zA-Z0-9\\s]", "");

        // Convert to lowercase (optional but recommended for proper counting)
        line = line.toLowerCase();

        // Tokenize the cleaned line
        StringTokenizer itr = new StringTokenizer(line);

        while (itr.hasMoreTokens()) {
            String token = itr.nextToken();
            context.write(new Text(token), new IntWritable(1));
        }
    }
  }

  public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {

    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {

        int sum = 0;
        for (IntWritable val : values) {
            sum += val.get();
        }
        context.write(key, new IntWritable(sum));
    }
  }

  public static void main(String[] args) throws Exception {

    if (args.length != 2) {
        System.err.println("Usage: WordCount <input path> <output path>");
        System.exit(2);
    }

    long startTime = System.currentTimeMillis();

    Configuration conf = new Configuration();

    Job job = Job.getInstance(conf, "WordCount Experiment");
    job.getConfiguration().setLong(
      "mapreduce.input.fileinputformat.split.maxsize",
      10485760   // 10MB for first experiment
    );
    job.setJarByClass(WordCount.class);

    job.setMapperClass(Map.class);
    job.setCombinerClass(Reduce.class);
    job.setReducerClass(Reduce.class);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);

    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));

    boolean success = job.waitForCompletion(true);

    long endTime = System.currentTimeMillis();

    System.out.println("Execution Time (ms): " + (endTime - startTime));

    System.exit(success ? 0 : 1);
  }
}
