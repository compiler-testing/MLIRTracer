module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<59x47x49xi8>) -> (tensor<i32>, tensor<1x47x49xi8>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tosa.reduce_product %arg2 {axis = 0 : i32} : (tensor<59x47x49xi8>) -> tensor<1x47x49xi8>
    %3 = tosa.arithmetic_right_shift %2, %2 {round = false} : (tensor<1x47x49xi8>, tensor<1x47x49xi8>) -> tensor<1x47x49xi8>
    return %1, %3 : tensor<i32>, tensor<1x47x49xi8>
  }
}
