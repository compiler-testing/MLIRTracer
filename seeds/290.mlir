module {
  func.func @main(%arg0: tensor<29x89x20xi1>, %arg1: tensor<29x89x1xi1>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<58x89x120xi1>, tensor<29x89x20xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<29x89x20xi1>, tensor<29x89x1xi1>) -> tensor<29x89x20xi1>
    %1 = tosa.exp %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.identity %1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<29x89x20xi1>, tensor<29x89x20xi1>) -> tensor<29x89x20xi1>
    %4 = tosa.bitwise_and %3, %0 : (tensor<29x89x20xi1>, tensor<29x89x20xi1>) -> tensor<29x89x20xi1>
    %5 = tosa.concat %4, %4 {axis = 2 : i32} : (tensor<29x89x20xi1>, tensor<29x89x20xi1>) -> tensor<29x89x40xi1>
    %6 = tosa.sigmoid %2 : (tensor<f32>) -> tensor<f32>
    %t_7 = tosa.const_shape {values = dense<[ 2, 1, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %7 = tosa.tile %5, %t_7 : (tensor<29x89x40xi1>, !tosa.shape<3>) -> tensor<58x89x120xi1>
    %8 = tosa.bitwise_xor %0, %0 : (tensor<29x89x20xi1>, tensor<29x89x20xi1>) -> tensor<29x89x20xi1>
    return %6, %7, %8 : tensor<f32>, tensor<58x89x120xi1>, tensor<29x89x20xi1>
  }
}
