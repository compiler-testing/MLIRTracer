module {
  func.func @main(%arg0: tensor<15x33xi1>, %arg1: tensor<1x1xi1>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<45x33xi1>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<15x33xi1>, tensor<1x1xi1>) -> tensor<15x33xi1>
    %t_1 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %0, %t_1 : (tensor<15x33xi1>, !tosa.shape<2>) -> tensor<45x33xi1>
    %2 = tosa.bitwise_or %1, %1 : (tensor<45x33xi1>, tensor<45x33xi1>) -> tensor<45x33xi1>
    %3 = tosa.sigmoid %arg2 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.bitwise_xor %2, %1 : (tensor<45x33xi1>, tensor<45x33xi1>) -> tensor<45x33xi1>
    %5 = tosa.bitwise_xor %4, %4 : (tensor<45x33xi1>, tensor<45x33xi1>) -> tensor<45x33xi1>
    return %3, %5 : tensor<f32>, tensor<45x33xi1>
  }
}
