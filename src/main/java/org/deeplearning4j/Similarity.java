package org.deeplearning4j;

import org.deeplearning4j.berkeley.StringUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.factory.Nd4j;


import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * Hello world!
 *
 */
public class Similarity {
    public static void main( String[] args ) throws  Exception {
        Nd4j.factory().setOrder('f');
        WordVectors vec = WordVectorSerializer.loadGoogleModel(new File(args[0]),Boolean.parseBoolean(args[1]));
        System.out.println( vec.similarity("ping-pong", "tennis"));
    }
}
