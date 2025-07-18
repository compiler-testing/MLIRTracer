module {
  func.func @main(%arg0: tensor<3xi16>, %arg1: tensor<3x66xf32>) -> (tensor<1xi16>, tensor<3x66xf32>) {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<3xi16>) -> tensor<1xi16>
    %1 = tosa.rsqrt %arg1 : (tensor<3x66xf32>) -> tensor<3x66xf32>
    %2 = tosa.identity %1 : (tensor<3x66xf32>) -> tensor<3x66xf32>
    return %0, %2 : tensor<1xi16>, tensor<3x66xf32>
  }
}
