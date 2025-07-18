module {
  func.func @main(%arg0: tensor<52x95xf32>) -> tensor<2x1x247x10xi1> {
    %0 = tosa.exp %arg0 : (tensor<52x95xf32>) -> tensor<52x95xf32>
    %1 = tosa.pow %0, %0 : (tensor<52x95xf32>, tensor<52x95xf32>) -> tensor<52x95xf32>
    %2 = tosa.greater_equal %1, %0 : (tensor<52x95xf32>, tensor<52x95xf32>) -> tensor<52x95xi1>
    %r_3 = tosa.const_shape {values = dense<[ 2, 1, 247, 10 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.reshape %2, %r_3 : (tensor<52x95xi1>, !tosa.shape<4>) -> tensor<2x1x247x10xi1>
    %4 = tosa.reverse %3 {axis = 0 : i32} : (tensor<2x1x247x10xi1>) -> tensor<2x1x247x10xi1>
    return %4 : tensor<2x1x247x10xi1>
  }
}
