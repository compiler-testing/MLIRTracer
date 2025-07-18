module {
  func.func @main(%arg0: tensor<59x1x15x78x4xi32>) -> tensor<i32> {
    %r_0 = tosa.const_shape {values = dense<[ 276120 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<59x1x15x78x4xi32>, !tosa.shape<1>) -> tensor<276120xi32>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<276120xi32>) -> tensor<i32>
    %2 = tosa.arithmetic_right_shift %1, %1 {round = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tosa.sub %2, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %3 : tensor<i32>
  }
}
