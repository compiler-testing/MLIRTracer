module {
  func.func @main(%arg0: tensor<30xf32>, %arg1: tensor<30xf32>, %arg2: tensor<60x54x20x96x29xf32>, %arg3: tensor<1x54x1x96x29xf32>) -> (tensor<1xi1>, tensor<60x54x20x96x29xi1>, tensor<60x54x20x96x29xf32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<30xf32>, tensor<30xf32>) -> tensor<30xi1>
    %1 = tosa.identity %0 : (tensor<30xi1>) -> tensor<30xi1>
    %2 = tosa.reduce_all %1 {axis = 0 : i32} : (tensor<30xi1>) -> tensor<1xi1>
    %3 = tosa.pow %arg2, %arg3 : (tensor<60x54x20x96x29xf32>, tensor<1x54x1x96x29xf32>) -> tensor<60x54x20x96x29xf32>
    %4 = tosa.greater %3, %3 : (tensor<60x54x20x96x29xf32>, tensor<60x54x20x96x29xf32>) -> tensor<60x54x20x96x29xi1>
    %5 = tosa.abs %3 : (tensor<60x54x20x96x29xf32>) -> tensor<60x54x20x96x29xf32>
    return %2, %4, %5 : tensor<1xi1>, tensor<60x54x20x96x29xi1>, tensor<60x54x20x96x29xf32>
  }
}
