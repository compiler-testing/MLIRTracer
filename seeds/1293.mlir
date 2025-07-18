module {
  func.func @main(%arg0: tensor<75x92xi1>, %arg1: tensor<1x92xi1>, %arg2: tensor<38x88x6x60xf32>, %arg3: tensor<1x1x6x60xf32>) -> (tensor<75x92xi1>, tensor<1x6xi32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<75x92xi1>, tensor<1x92xi1>) -> tensor<75x92xi1>
    %1 = tosa.identity %0 : (tensor<75x92xi1>) -> tensor<75x92xi1>
    %2 = tosa.pow %arg2, %arg3 : (tensor<38x88x6x60xf32>, tensor<1x1x6x60xf32>) -> tensor<38x88x6x60xf32>
    %3 = tosa.argmax %2 {axis = 1 : i32} : (tensor<38x88x6x60xf32>) -> tensor<38x6x60xi32>
    %4 = tosa.logical_and %1, %0 : (tensor<75x92xi1>, tensor<75x92xi1>) -> tensor<75x92xi1>
    %5 = tosa.argmax %3 {axis = 2 : i32} : (tensor<38x6x60xi32>) -> tensor<38x6xi32>
    %6 = tosa.reduce_min %5 {axis = 0 : i32} : (tensor<38x6xi32>) -> tensor<1x6xi32>
    return %4, %6 : tensor<75x92xi1>, tensor<1x6xi32>
  }
}
