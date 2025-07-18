module {
  func.func @main(%arg0: tensor<85x67x9xi16>) -> tensor<1x1xi32> {
    %0 = tosa.reduce_min %arg0 {axis = 2 : i32} : (tensor<85x67x9xi16>) -> tensor<85x67x1xi16>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<85x67x1xi16>) -> tensor<67x1xi32>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<67x1xi32>) -> tensor<1x1xi32>
    return %2 : tensor<1x1xi32>
  }
}
