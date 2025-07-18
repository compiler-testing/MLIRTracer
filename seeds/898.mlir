module {
  func.func @main(%arg0: tensor<72x7x27x33xi1>) -> tensor<72x189x1xi1> {
    %0 = tosa.reduce_all %arg0 {axis = 3 : i32} : (tensor<72x7x27x33xi1>) -> tensor<72x7x27x1xi1>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<72x7x27x1xi1>) -> tensor<72x7x27x1xi1>
    %r_2 = tosa.const_shape {values = dense<[ 72, 189, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.reshape %1, %r_2 : (tensor<72x7x27x1xi1>, !tosa.shape<3>) -> tensor<72x189x1xi1>
    return %2 : tensor<72x189x1xi1>
  }
}
