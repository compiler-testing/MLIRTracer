module {
  func.func @main(%arg0: tensor<84x22x89xi64>, %arg1: tensor<1x22x1xi64>, %arg2: tensor<66x96x27xi1>, %arg3: tensor<66x1x27xi1>, %arg4: tensor<f32>) -> (tensor<84x22x89xi64>, tensor<f32>, tensor<1x96x27xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<84x22x89xi64>, tensor<1x22x1xi64>) -> tensor<84x22x89xi64>
    %1 = tosa.abs %0 : (tensor<84x22x89xi64>) -> tensor<84x22x89xi64>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<66x96x27xi1>, tensor<66x1x27xi1>) -> tensor<66x96x27xi1>
    %3 = tosa.sigmoid %arg4 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.ceil %3 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<66x96x27xi1>) -> tensor<1x96x27xi1>
    %6 = tosa.bitwise_xor %5, %5 : (tensor<1x96x27xi1>, tensor<1x96x27xi1>) -> tensor<1x96x27xi1>
    return %1, %4, %6 : tensor<84x22x89xi64>, tensor<f32>, tensor<1x96x27xi1>
  }
}
