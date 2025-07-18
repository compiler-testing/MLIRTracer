module {
  func.func @main(%arg0: tensor<33x84xi1>, %arg1: tensor<55x50x12x94x82x51xf32>, %arg2: tensor<87x35x86x92xf32>, %arg3: tensor<81x47x98x94xf32>, %arg4: tensor<81xf32>) -> (tensor<55x50x12x94x82x51xf32>, tensor<87x119x270x81xf32>, tensor<33x84xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<33x84xi1>) -> tensor<33x84xi1>
    %1 = tosa.clz %0 : (tensor<33x84xi1>) -> tensor<33x84xi1>
    %2 = tosa.log %arg1 : (tensor<55x50x12x94x82x51xf32>) -> tensor<55x50x12x94x82x51xf32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 87, 119, 270, 81>} : (tensor<87x35x86x92xf32>, tensor<81x47x98x94xf32>, tensor<81xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<87x119x270x81xf32>
    %4 = tosa.add %3, %3 : (tensor<87x119x270x81xf32>, tensor<87x119x270x81xf32>) -> tensor<87x119x270x81xf32>
    %5 = tosa.bitwise_and %1, %1 : (tensor<33x84xi1>, tensor<33x84xi1>) -> tensor<33x84xi1>
    return %2, %4, %5 : tensor<55x50x12x94x82x51xf32>, tensor<87x119x270x81xf32>, tensor<33x84xi1>
  }
}
