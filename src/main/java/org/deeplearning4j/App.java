package org.deeplearning4j;

import org.deeplearning4j.berkeley.StringUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.factory.Nd4j;


import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * Hello world!
 *
 */
public class App {
    public static void main( String[] args ) throws  Exception {
        Nd4j.factory().setOrder('f');
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(new File(args[0]),Boolean.parseBoolean(args[1]));
        List<String> words = StringUtils.split(args[2],",");
        for(String s : words) {
            System.out.println("Vector for word " + Arrays.toString(vec.getWordVector(s)));
            System.out.println("Words nearest " + vec.wordsNearest(s,20));

        }
    }
}
