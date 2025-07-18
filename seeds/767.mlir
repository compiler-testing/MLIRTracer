module {
  func.func @main(%arg0: tensor<13x56x50x5x35x75xi64>, %arg1: tensor<13x1x50x1x1x75xi64>, %arg2: tensor<32xi1>, %arg3: tensor<86x88x44xf32>) -> (tensor<1xi1>, tensor<13x56x100x5x35x75xi64>, tensor<86x88x44xf32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<13x56x50x5x35x75xi64>, tensor<13x1x50x1x1x75xi64>) -> tensor<13x56x50x5x35x75xi64>
    %1 = tosa.reduce_all %arg2 {axis = 0 : i32} : (tensor<32xi1>) -> tensor<1xi1>
    %2 = tosa.logical_right_shift %0, %0 : (tensor<13x56x50x5x35x75xi64>, tensor<13x56x50x5x35x75xi64>) -> tensor<13x56x50x5x35x75xi64>
    %3 = tosa.concat %2, %0 {axis = 2 : i32} : (tensor<13x56x50x5x35x75xi64>, tensor<13x56x50x5x35x75xi64>) -> tensor<13x56x100x5x35x75xi64>
    %4 = tosa.floor %arg3 : (tensor<86x88x44xf32>) -> tensor<86x88x44xf32>
    return %1, %3, %4 : tensor<1xi1>, tensor<13x56x100x5x35x75xi64>, tensor<86x88x44xf32>
  }
}
