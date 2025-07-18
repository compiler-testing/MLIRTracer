module {
  func.func @main(%arg0: tensor<19x93x20x97x97x34xi8>, %arg1: tensor<9xi1>, %arg2: tensor<9xi1>, %arg3: tensor<5x85x11x57x2xf32>) -> (tensor<19x93x20x97x97x34xi8>, tensor<1xi1>, tensor<1xi1>, tensor<5x85x11x57x2xf32>) {
    %0 = tosa.clz %arg0 : (tensor<19x93x20x97x97x34xi8>) -> tensor<19x93x20x97x97x34xi8>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %2 = tosa.reduce_all %1 {axis = 0 : i32} : (tensor<9xi1>) -> tensor<1xi1>
    %3 = tosa.reverse %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<9xi1>) -> tensor<1xi1>
    %5 = tosa.floor %arg3 : (tensor<5x85x11x57x2xf32>) -> tensor<5x85x11x57x2xf32>
    return %0, %3, %4, %5 : tensor<19x93x20x97x97x34xi8>, tensor<1xi1>, tensor<1xi1>, tensor<5x85x11x57x2xf32>
  }
}
