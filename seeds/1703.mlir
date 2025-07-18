module {
  func.func @main(%arg0: tensor<64xi32>) -> tensor<64xi32> {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<64xi32>) -> tensor<64xi32>
    %1 = tosa.bitwise_not %0 : (tensor<64xi32>) -> tensor<64xi32>
    %r_2 = tosa.const_shape {values = dense<[ 64 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.reshape %1, %r_2 : (tensor<64xi32>, !tosa.shape<1>) -> tensor<64xi32>
    return %2 : tensor<64xi32>
  }
}
