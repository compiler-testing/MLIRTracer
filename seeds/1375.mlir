module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<2xi8>, %arg2: tensor<98x71x48x89xi1>) -> (tensor<f32>, tensor<2xi8>, tensor<1x71x48x89xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.bitwise_not %arg1 : (tensor<2xi8>) -> tensor<2xi8>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<2xi8>, tensor<2xi8>) -> tensor<2xi8>
    %3 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<98x71x48x89xi1>) -> tensor<1x71x48x89xi1>
    return %0, %2, %3 : tensor<f32>, tensor<2xi8>, tensor<1x71x48x89xi1>
  }
}
