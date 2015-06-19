package org.deeplearning4j;

import org.deeplearning4j.berkeley.StringUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.File;
import java.util.List;

/**
 * Hello world!
 *
 */
public class App {
    public static void main( String[] args ) throws  Exception {
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(new File(args[0]),Boolean.parseBoolean(args[1]));
        List<String> words = StringUtils.split(args[2],",");

        for(String s : words) {
            System.out.println(vec.getWordVector(s));
        }
    }
}
