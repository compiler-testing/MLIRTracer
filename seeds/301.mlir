module {
  func.func @main(%arg0: tensor<92x64xi1>, %arg1: tensor<14xf32>) -> (tensor<1x1xi1>, tensor<14xf32>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<92x64xi1>) -> tensor<1x64xi1>
    %1 = tosa.sub %0, %0 : (tensor<1x64xi1>, tensor<1x64xi1>) -> tensor<1x64xi1>
    %2 = tosa.add %1, %1 : (tensor<1x64xi1>, tensor<1x64xi1>) -> tensor<1x64xi1>
    %3 = tosa.reduce_any %2 {axis = 0 : i32} : (tensor<1x64xi1>) -> tensor<1x64xi1>
    %4 = tosa.reduce_sum %3 {axis = 1 : i32} : (tensor<1x64xi1>) -> tensor<1x1xi1>
    %5 = tosa.sigmoid %arg1 : (tensor<14xf32>) -> tensor<14xf32>
    return %4, %5 : tensor<1x1xi1>, tensor<14xf32>
  }
}
