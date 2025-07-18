module {
  func.func @main(%arg0: tensor<81x53xf32>, %arg1: tensor<11x72x51x40xi1>) -> (tensor<81x1x53x1xf32>, tensor<11x1x51x40xi1>) {
    %0 = tosa.tanh %arg0 : (tensor<81x53xf32>) -> tensor<81x53xf32>
    %1 = tosa.ceil %0 : (tensor<81x53xf32>) -> tensor<81x53xf32>
    %r_2 = tosa.const_shape {values = dense<[ 81, 1, 53, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.reshape %1, %r_2 : (tensor<81x53xf32>, !tosa.shape<4>) -> tensor<81x1x53x1xf32>
    %3 = tosa.sigmoid %2 : (tensor<81x1x53x1xf32>) -> tensor<81x1x53x1xf32>
    %4 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<11x72x51x40xi1>) -> tensor<11x1x51x40xi1>
    %5 = tosa.abs %4 : (tensor<11x1x51x40xi1>) -> tensor<11x1x51x40xi1>
    return %3, %5 : tensor<81x1x53x1xf32>, tensor<11x1x51x40xi1>
  }
}
