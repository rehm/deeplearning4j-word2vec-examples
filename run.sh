#!/bin/bash
java -Xmx10g -Xms10g -cp target/deeplearning4j-word2vec-test-1.0-SNAPSHOT.jar GoogleNews-vectors-negative300.bin.gz true
