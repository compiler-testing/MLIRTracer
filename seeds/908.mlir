module {
  func.func @main(%arg0: tensor<6xi16>, %arg1: tensor<6xi16>, %arg2: tensor<25x88xf32>) -> (tensor<8xi16>, tensor<1xi16>, tensor<i32>, tensor<25x88xf32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<6xi16>, tensor<6xi16>) -> tensor<6xi16>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<6xi16>) -> tensor<i32>
    %s_2_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_2_size = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.slice %0, %s_2_start, %s_2_size : (tensor<6xi16>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<8xi16>
    %3 = tosa.bitwise_xor %1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<6xi16>) -> tensor<1xi16>
    %5 = tosa.arithmetic_right_shift %3, %1 {round = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = tosa.bitwise_or %5, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = tosa.tanh %arg2 : (tensor<25x88xf32>) -> tensor<25x88xf32>
    %in_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %8 = tosa.negate %7, %in_zp_8, %out_zp_8 : (tensor<25x88xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<25x88xf32>
    return %2, %4, %6, %8 : tensor<8xi16>, tensor<1xi16>, tensor<i32>, tensor<25x88xf32>
  }
}
