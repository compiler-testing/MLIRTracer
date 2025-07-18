module {
  func.func @main(%arg0: tensor<6x9xi8>, %arg1: tensor<90x11x92xi1>) -> (tensor<1x1xi8>, tensor<90x2x92xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<6x9xi8>) -> tensor<6x1xi8>
    %1 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<6x1xi8>) -> tensor<1x1xi8>
    %2 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<90x11x92xi1>) -> tensor<90x1x92xi1>
    %3 = tosa.concat %2, %2 {axis = 1 : i32} : (tensor<90x1x92xi1>, tensor<90x1x92xi1>) -> tensor<90x2x92xi1>
    return %1, %3 : tensor<1x1xi8>, tensor<90x2x92xi1>
  }
}
