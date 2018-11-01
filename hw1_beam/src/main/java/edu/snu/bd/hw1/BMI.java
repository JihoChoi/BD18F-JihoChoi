package edu.snu.bd.hw1;

import org.apache.beam.sdk.extensions.sql.BeamSqlUdf;


public class BMI implements BeamSqlUdf {
    public static Double eval(Double height, Double weight) {
    // public static Double eval(Double height) {
        // (cm) -> (m)
        double bmi = weight / (height/100 * height/100);
        return bmi;
    }
}
//
// public static class BMIFn implements SerializableFunction<Integer, Integer> {
//     @Override
//     public Integer apply(Integer input) {
//         return input * input * input;
//     }
// }
