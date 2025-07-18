module {
  func.func @main(%arg0: tensor<75x15xi32>, %arg1: tensor<13x15xi32>) -> tensor<176x1xi32> {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<75x15xi32>, tensor<13x15xi32>) -> tensor<88x15xi32>
    %1 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<88x15xi32>) -> tensor<88x1xi32>
    %2 = tosa.concat %1, %1 {axis = 0 : i32} : (tensor<88x1xi32>, tensor<88x1xi32>) -> tensor<176x1xi32>
    return %2 : tensor<176x1xi32>
  }
}
