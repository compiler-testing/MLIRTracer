module {
  func.func @main(%arg0: tensor<60x40xi1>, %arg1: tensor<64x47x74x35x97x98xf32>) -> (tensor<64x47x74x35x97x98xf32>, tensor<1x1xi1>, tensor<1xi32>) {
    %0 = tosa.reduce_min %arg0 {axis = 1 : i32} : (tensor<60x40xi1>) -> tensor<60x1xi1>
    %1 = tosa.ceil %arg1 : (tensor<64x47x74x35x97x98xf32>) -> tensor<64x47x74x35x97x98xf32>
    %2 = tosa.maximum %1, %1 : (tensor<64x47x74x35x97x98xf32>, tensor<64x47x74x35x97x98xf32>) -> tensor<64x47x74x35x97x98xf32>
    %3 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<60x1xi1>) -> tensor<1x1xi1>
    %4 = tosa.argmax %0 {axis = 0 : i32} : (tensor<60x1xi1>) -> tensor<1xi32>
    return %2, %3, %4 : tensor<64x47x74x35x97x98xf32>, tensor<1x1xi1>, tensor<1xi32>
  }
}
