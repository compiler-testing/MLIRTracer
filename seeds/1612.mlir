module {
  func.func @main(%arg0: tensor<67xf32>, %arg1: tensor<1xf32>) -> tensor<1x1x1xi1> {
    %0 = tosa.greater %arg0, %arg1 : (tensor<67xf32>, tensor<1xf32>) -> tensor<67xi1>
    %1 = tosa.reduce_all %0 {axis = 0 : i32} : (tensor<67xi1>) -> tensor<1xi1>
    %r_2 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.reshape %1, %r_2 : (tensor<1xi1>, !tosa.shape<3>) -> tensor<1x1x1xi1>
    %3 = tosa.bitwise_not %2 : (tensor<1x1x1xi1>) -> tensor<1x1x1xi1>
    return %3 : tensor<1x1x1xi1>
  }
}
