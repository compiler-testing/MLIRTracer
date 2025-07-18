module {
  func.func @main(%arg0: tensor<55xf32>, %arg1: tensor<80x48xi1>, %arg2: tensor<86x76x51x90xi32>, %arg3: tensor<86x1x1x1xi32>) -> (tensor<55xf32>, tensor<1x48xi1>, tensor<86x76x51x90xi32>) {
    %0 = tosa.ceil %arg0 : (tensor<55xf32>) -> tensor<55xf32>
    %1 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<80x48xi1>) -> tensor<1x48xi1>
    %2 = tosa.reduce_all %1 {axis = 0 : i32} : (tensor<1x48xi1>) -> tensor<1x48xi1>
    %3 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<1x48xi1>) -> tensor<1x48xi1>
    %4 = tosa.logical_xor %3, %2 : (tensor<1x48xi1>, tensor<1x48xi1>) -> tensor<1x48xi1>
    %5 = tosa.intdiv %arg2, %arg3 : (tensor<86x76x51x90xi32>, tensor<86x1x1x1xi32>) -> tensor<86x76x51x90xi32>
    return %0, %4, %5 : tensor<55xf32>, tensor<1x48xi1>, tensor<86x76x51x90xi32>
  }
}
