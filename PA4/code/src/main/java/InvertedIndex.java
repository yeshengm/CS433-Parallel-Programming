import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;

public class InvertedIndex {
    /*
     * Divide input Text to tokens and remove all the punctuations in the token
     * Output format: (word#filename, 1)
     */
    public static class InvertedIndexMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            // get file name of input split
            String fileName = fileSplit.getPath().getName();
            Text word = new Text();
            StringTokenizer itr = new StringTokenizer(value.toString());
            for (; itr.hasMoreTokens(); ) {
                word.set(itr.nextToken().replaceAll("\\p{Punct}+", "") + "#" + fileName);
                // write (word#filename, 1)
                context.write(word, one);
            }
        }
    }

    /* A combiner to reduce comunication overhead */
    public static class SumCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    /* Partitioner: put key-value pairs with the same word to the same reducer */
    public static class WordPartitioner extends HashPartitioner<Text, IntWritable> {
        public int getPartition(Text key, IntWritable value, int numReduceTasks) {
            String term = key.toString().split("#")[0];
            return super.getPartition(new Text(term), value, numReduceTasks);
        }
    }

    public static class InvertedIndexReducer extends Reducer<Text, IntWritable, Text, Text> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            String term = key.toString().split("#")[0];
            if (!term.matches("[a-zA-z0-9]+"))
                return;
            String filename = key.toString().split("#")[1];
            for (IntWritable i : values) {
                context.write(new Text(term), new Text(filename + ":" + i));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        Job job = new Job(conf, "inverted index");
        job.setJarByClass(InvertedIndex.class);
        job.setMapperClass(InvertedIndexMapper.class);
        job.setCombinerClass(SumCombiner.class);
        job.setReducerClass(InvertedIndexReducer.class);
        job.setNumReduceTasks(1);
        job.setPartitionerClass(WordPartitioner.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path("hdfs://localhost:9000/input"));
        FileOutputFormat.setOutputPath(job, new Path("hdfs://localhost:9000/output"));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}