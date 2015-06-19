package org.deeplearning4j;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class LargeBuffer {
    public static void main(String[] args) {
        DataBuffer b = Nd4j.createBuffer(Integer.parseInt(args[0]));
        System.out.println("Created buffer of size " + b.length());
    }


}
