module {
  func.func @main(%arg0: tensor<87x51x96x97x57xf32>, %arg1: tensor<65xi1>) -> (tensor<87x51x96x97x57xf32>, tensor<1xi1>) {
    %0 = tosa.identity %arg0 : (tensor<87x51x96x97x57xf32>) -> tensor<87x51x96x97x57xf32>
    %1 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<65xi1>) -> tensor<1xi1>
    return %0, %1 : tensor<87x51x96x97x57xf32>, tensor<1xi1>
  }
}
