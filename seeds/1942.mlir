module {
  func.func @main(%arg0: tensor<18x15x97x43x31x24xi1>, %arg1: tensor<1x1x97x43x1x24xi1>, %arg2: tensor<97x93x92x33xf32>) -> (tensor<18x15x97x43x31x24xi1>, tensor<97x93x92x1xf32>, tensor<97x93x92x33xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<18x15x97x43x31x24xi1>, tensor<1x1x97x43x1x24xi1>) -> tensor<18x15x97x43x31x24xi1>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<18x15x97x43x31x24xi1>, tensor<18x15x97x43x31x24xi1>) -> tensor<18x15x97x43x31x24xi1>
    %2 = tosa.logical_xor %1, %1 : (tensor<18x15x97x43x31x24xi1>, tensor<18x15x97x43x31x24xi1>) -> tensor<18x15x97x43x31x24xi1>
    %3 = tosa.exp %arg2 : (tensor<97x93x92x33xf32>) -> tensor<97x93x92x33xf32>
    %4 = tosa.equal %3, %3 : (tensor<97x93x92x33xf32>, tensor<97x93x92x33xf32>) -> tensor<97x93x92x33xi1>
    %5 = tosa.reduce_min %3 {axis = 3 : i32} : (tensor<97x93x92x33xf32>) -> tensor<97x93x92x1xf32>
    %6 = tosa.logical_left_shift %4, %4 : (tensor<97x93x92x33xi1>, tensor<97x93x92x33xi1>) -> tensor<97x93x92x33xi1>
    return %2, %5, %6 : tensor<18x15x97x43x31x24xi1>, tensor<97x93x92x1xf32>, tensor<97x93x92x33xi1>
  }
}
