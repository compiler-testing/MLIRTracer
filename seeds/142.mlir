module {
  func.func @main(%arg0: tensor<84x32xi64>, %arg1: tensor<1x1xi64>, %arg2: tensor<34x71x30x98x83x78xi1>, %arg3: tensor<1x71x30x98x83x1xi1>, %arg4: tensor<86x48x71x60x82x28xf32>, %arg5: tensor<86x1x71x60x82x28xf32>) -> (tensor<1x32xi64>, tensor<34x71x30x196x83x78xi1>, tensor<86x48x71x60x82x28xf32>, tensor<86x48x71x60x82x28xf32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<84x32xi64>, tensor<1x1xi64>) -> tensor<84x32xi64>
    %1 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<84x32xi64>) -> tensor<1x32xi64>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<34x71x30x98x83x78xi1>, tensor<1x71x30x98x83x1xi1>) -> tensor<34x71x30x98x83x78xi1>
    %3 = tosa.logical_right_shift %1, %1 : (tensor<1x32xi64>, tensor<1x32xi64>) -> tensor<1x32xi64>
    %4 = tosa.concat %2, %2 {axis = 3 : i32} : (tensor<34x71x30x98x83x78xi1>, tensor<34x71x30x98x83x78xi1>) -> tensor<34x71x30x196x83x78xi1>
    %5 = tosa.pow %arg4, %arg5 : (tensor<86x48x71x60x82x28xf32>, tensor<86x1x71x60x82x28xf32>) -> tensor<86x48x71x60x82x28xf32>
    %6 = tosa.tanh %5 : (tensor<86x48x71x60x82x28xf32>) -> tensor<86x48x71x60x82x28xf32>
    %7 = tosa.minimum %5, %6 : (tensor<86x48x71x60x82x28xf32>, tensor<86x48x71x60x82x28xf32>) -> tensor<86x48x71x60x82x28xf32>
    %8 = tosa.rsqrt %5 : (tensor<86x48x71x60x82x28xf32>) -> tensor<86x48x71x60x82x28xf32>
    return %3, %4, %7, %8 : tensor<1x32xi64>, tensor<34x71x30x196x83x78xi1>, tensor<86x48x71x60x82x28xf32>, tensor<86x48x71x60x82x28xf32>
  }
}
