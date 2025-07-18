module {
  func.func @main(%arg0: tensor<84xi8>, %arg1: tensor<1xi8>, %arg2: tensor<68x72x23x6xf32>, %arg3: tensor<50x66x21x6xf32>, %arg4: tensor<50xf32>) -> (tensor<84xi1>, tensor<1x210x47x50xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<84xi8>, tensor<1xi8>) -> tensor<84xi8>
    %1 = tosa.greater_equal %0, %0 : (tensor<84xi8>, tensor<84xi8>) -> tensor<84xi1>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 68, 210, 47, 50>} : (tensor<68x72x23x6xf32>, tensor<50x66x21x6xf32>, tensor<50xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<68x210x47x50xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<84xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<84xi1>
    %4 = tosa.maximum %2, %2 : (tensor<68x210x47x50xf32>, tensor<68x210x47x50xf32>) -> tensor<68x210x47x50xf32>
    %5 = tosa.reduce_product %4 {axis = 0 : i32} : (tensor<68x210x47x50xf32>) -> tensor<1x210x47x50xf32>
    %6 = tosa.add %5, %5 : (tensor<1x210x47x50xf32>, tensor<1x210x47x50xf32>) -> tensor<1x210x47x50xf32>
    return %3, %6 : tensor<84xi1>, tensor<1x210x47x50xf32>
  }
}
