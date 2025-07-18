module {
  func.func @main(%arg0: tensor<79x28x29x16xf32>, %arg1: tensor<79x9xi16>, %arg2: tensor<1x1xi16>) -> (tensor<79x28x29x1xf32>, tensor<79x9xi16>) {
    %0 = tosa.tanh %arg0 : (tensor<79x28x29x16xf32>) -> tensor<79x28x29x16xf32>
    %1 = tosa.pow %0, %0 : (tensor<79x28x29x16xf32>, tensor<79x28x29x16xf32>) -> tensor<79x28x29x16xf32>
    %2 = tosa.logical_right_shift %arg1, %arg2 : (tensor<79x9xi16>, tensor<1x1xi16>) -> tensor<79x9xi16>
    %3 = tosa.reduce_product %1 {axis = 3 : i32} : (tensor<79x28x29x16xf32>) -> tensor<79x28x29x1xf32>
    %4 = tosa.abs %2 : (tensor<79x9xi16>) -> tensor<79x9xi16>
    return %3, %4 : tensor<79x28x29x1xf32>, tensor<79x9xi16>
  }
}
