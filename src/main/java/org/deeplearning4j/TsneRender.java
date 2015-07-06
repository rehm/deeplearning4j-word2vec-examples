package org.deeplearning4j;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;

import java.io.File;
import java.util.ArrayList;

/**
 *  * @author Adam Gibson
 *   */
public class TsneRender {
    public static void main(String[] args) throws Exception {
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .theta(0.5).learningRate(500).setMaxIter(1000)
                .build();
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(new File(args[0]), true);
        InMemoryLookupTable table = (InMemoryLookupTable) vec.lookupTable();
        tsne.plot(table.getSyn0(),2,new ArrayList<>(vec.vocab().words()));

    }

}

