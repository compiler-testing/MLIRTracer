module {
  func.func @main(%arg0: tensor<81x38xi16>, %arg1: tensor<94x62x14x1x29x60xf32>) -> (tensor<81x38xi16>, tensor<94x62x14x1x29x60xf32>) {
    %0 = tosa.clz %arg0 : (tensor<81x38xi16>) -> tensor<81x38xi16>
    %1 = tosa.reverse %0 {axis = 1 : i32} : (tensor<81x38xi16>) -> tensor<81x38xi16>
    %2 = tosa.tanh %arg1 : (tensor<94x62x14x1x29x60xf32>) -> tensor<94x62x14x1x29x60xf32>
    %3 = tosa.exp %2 : (tensor<94x62x14x1x29x60xf32>) -> tensor<94x62x14x1x29x60xf32>
    return %1, %3 : tensor<81x38xi16>, tensor<94x62x14x1x29x60xf32>
  }
}
