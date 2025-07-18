module {
  func.func @main(%arg0: tensor<86x78x84x84xi8>, %arg1: tensor<86x1x84x1xi8>, %arg2: tensor<36x91x37x19x31x93xf32>) -> (tensor<86x84x78x84xi1>, tensor<36x91x37x19x31x93xf32>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<86x78x84x84xi8>, tensor<86x1x84x1xi8>) -> tensor<86x78x84x84xi8>
    %1 = tosa.reverse %0 {axis = 3 : i32} : (tensor<86x78x84x84xi8>) -> tensor<86x78x84x84xi8>
    %2 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 0, 2, 1, 3>} : (tensor<86x78x84x84xi8>) -> tensor<86x84x78x84xi8>
    %4 = tosa.maximum %3, %3 : (tensor<86x84x78x84xi8>, tensor<86x84x78x84xi8>) -> tensor<86x84x78x84xi8>
    %5 = tosa.equal %4, %3 : (tensor<86x84x78x84xi8>, tensor<86x84x78x84xi8>) -> tensor<86x84x78x84xi1>
    %6 = tosa.arithmetic_right_shift %5, %5 {round = false} : (tensor<86x84x78x84xi1>, tensor<86x84x78x84xi1>) -> tensor<86x84x78x84xi1>
    %7 = tosa.logical_xor %6, %5 : (tensor<86x84x78x84xi1>, tensor<86x84x78x84xi1>) -> tensor<86x84x78x84xi1>
    %8 = tosa.tanh %arg2 : (tensor<36x91x37x19x31x93xf32>) -> tensor<36x91x37x19x31x93xf32>
    %9 = tosa.arithmetic_right_shift %7, %6 {round = false} : (tensor<86x84x78x84xi1>, tensor<86x84x78x84xi1>) -> tensor<86x84x78x84xi1>
    %10 = tosa.floor %8 : (tensor<36x91x37x19x31x93xf32>) -> tensor<36x91x37x19x31x93xf32>
    %11 = tosa.pow %10, %8 : (tensor<36x91x37x19x31x93xf32>, tensor<36x91x37x19x31x93xf32>) -> tensor<36x91x37x19x31x93xf32>
    return %9, %11 : tensor<86x84x78x84xi1>, tensor<36x91x37x19x31x93xf32>
  }
}
