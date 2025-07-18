module {
  func.func @main(%arg0: tensor<81x53x94x82x36xi16>, %arg1: tensor<1x1x1x82x1xi16>, %arg2: tensor<62x52xf32>) -> (tensor<81x53x94x82x36xi16>, tensor<62x52xf32>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<81x53x94x82x36xi16>, tensor<1x1x1x82x1xi16>) -> tensor<81x53x94x82x36xi16>
    %1 = tosa.bitwise_or %0, %0 : (tensor<81x53x94x82x36xi16>, tensor<81x53x94x82x36xi16>) -> tensor<81x53x94x82x36xi16>
    %2 = tosa.reciprocal %arg2 : (tensor<62x52xf32>) -> tensor<62x52xf32>
    %3 = tosa.floor %2 : (tensor<62x52xf32>) -> tensor<62x52xf32>
    return %1, %3 : tensor<81x53x94x82x36xi16>, tensor<62x52xf32>
  }
}
