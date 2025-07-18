module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<96x47x28xi32>, %arg3: tensor<1x47x28xi32>, %arg4: tensor<49xi1>) -> (tensor<i1>, tensor<96x47x28xi32>, tensor<1xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.logical_or %1, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.intdiv %arg2, %arg3 : (tensor<96x47x28xi32>, tensor<1x47x28xi32>) -> tensor<96x47x28xi32>
    %4 = tosa.reduce_all %arg4 {axis = 0 : i32} : (tensor<49xi1>) -> tensor<1xi1>
    return %2, %3, %4 : tensor<i1>, tensor<96x47x28xi32>, tensor<1xi1>
  }
}
