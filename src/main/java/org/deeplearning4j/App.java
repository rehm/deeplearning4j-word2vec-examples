package org.deeplearning4j;

import org.deeplearning4j.berkeley.StringUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

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
            INDArray vecFirst = vec.getWordVectorMatrix(s);
            INDArray noOffset = vecFirst.dup();
            for(String s2 : words) {
                INDArray vecSecond = vec.getWordVectorMatrix(s2);
                INDArray noOffsetSecond = vecSecond.dup();
                System.out.println("Offset " + s + " " + s2 + " " + Transforms.cosineSim(vecFirst,vecSecond));
                System.out.println("No Offset " + s + " " + s2 + " " + Transforms.cosineSim(noOffset,noOffsetSecond));
            }

        }
    }
}
