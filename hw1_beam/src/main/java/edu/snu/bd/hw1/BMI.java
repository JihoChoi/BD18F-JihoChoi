package edu.snu.bd.hw1;

import org.apache.beam.sdk.extensions.sql.BeamSqlUdf;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/*
    ND4J is a scientific computing library for JVM. (Similar to numPy in Python)
    + DL4J is a deep learning library for JVM.

    The example is not very adequate since the given calculation
    is already too simple. However, this is the idea of how scientific
    computing can be done in a distributed manner with GPU usage.

    - Jiho Choi
*/

public class BMI implements BeamSqlUdf {
    public static Double eval(Double height, Double weight) {

        // (cm) -> (m)
        // double bmi = weight / (height/100 * height/100);

        // INDArray nd = Nd4j.zeros(1, 2); // row, column
        // nd.addi(weight);

        INDArray nd = Nd4j.create(new double[]{weight, weight},new int[]{1,2});
        nd.divi(height / 100 * height / 100);

        return nd.getDouble(1,1);
        // return bmi;
    }
}
//
// public static class BMIFn implements SerializableFunction<Integer, Integer> {
//     @Override
//     public Integer apply(Integer input) {
//         return input * input * input;
//     }
// }
