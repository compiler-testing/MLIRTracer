module {
  func.func @main(%arg0: tensor<84x4x51xi64>, %arg1: tensor<65x12x61xi32>, %arg2: tensor<65x1x61xi32>, %arg3: tensor<4x97x94x44x54x31xf32>) -> (tensor<84x1x1xi64>, tensor<4x97x94x44x54x31xf32>, tensor<65x12x61xi32>) {
    %0 = tosa.reduce_min %arg0 {axis = 2 : i32} : (tensor<84x4x51xi64>) -> tensor<84x4x1xi64>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<84x4x1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<84x4x1xi64>
    %2 = tosa.reduce_min %1 {axis = 1 : i32} : (tensor<84x4x1xi64>) -> tensor<84x1x1xi64>
    %3 = tosa.intdiv %arg1, %arg2 : (tensor<65x12x61xi32>, tensor<65x1x61xi32>) -> tensor<65x12x61xi32>
    %4 = tosa.sigmoid %arg3 : (tensor<4x97x94x44x54x31xf32>) -> tensor<4x97x94x44x54x31xf32>
    %5 = tosa.abs %3 : (tensor<65x12x61xi32>) -> tensor<65x12x61xi32>
    return %2, %4, %5 : tensor<84x1x1xi64>, tensor<4x97x94x44x54x31xf32>, tensor<65x12x61xi32>
  }
}
