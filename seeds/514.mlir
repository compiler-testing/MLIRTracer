module {
  func.func @main(%arg0: tensor<57x87x94xi1>, %arg1: tensor<57x87x94xi1>, %arg2: tensor<60x54xf32>) -> (tensor<1x87x94xi1>, tensor<57x87x94xi1>, tensor<60x1xi1>, tensor<60x54xf32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<57x87x94xi1>, tensor<57x87x94xi1>) -> tensor<57x87x94xi1>
    %1 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<57x87x94xi1>) -> tensor<1x87x94xi1>
    %2 = tosa.log %arg2 : (tensor<60x54xf32>) -> tensor<60x54xf32>
    %3 = tosa.sigmoid %2 : (tensor<60x54xf32>) -> tensor<60x54xf32>
    %4 = tosa.identity %2 : (tensor<60x54xf32>) -> tensor<60x54xf32>
    %5 = tosa.maximum %3, %3 : (tensor<60x54xf32>, tensor<60x54xf32>) -> tensor<60x54xf32>
    %6 = tosa.reduce_sum %4 {axis = 1 : i32} : (tensor<60x54xf32>) -> tensor<60x1xf32>
    %7 = tosa.minimum %6, %6 : (tensor<60x1xf32>, tensor<60x1xf32>) -> tensor<60x1xf32>
    %8 = tosa.logical_not %0 : (tensor<57x87x94xi1>) -> tensor<57x87x94xi1>
    %9 = tosa.equal %7, %6 : (tensor<60x1xf32>, tensor<60x1xf32>) -> tensor<60x1xi1>
    %10 = tosa.maximum %5, %5 : (tensor<60x54xf32>, tensor<60x54xf32>) -> tensor<60x54xf32>
    return %1, %8, %9, %10 : tensor<1x87x94xi1>, tensor<57x87x94xi1>, tensor<60x1xi1>, tensor<60x54xf32>
  }
}
