module {
  func.func @main(%arg0: tensor<91x30xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<55x9x99x6x81xi1>, %arg3: tensor<55x1x99x6x81xi1>) -> (tensor<91x30xf32>, tensor<55x9x99x6x81xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<91x30xf32>, tensor<1x1xf32>) -> tensor<91x30xf32>
    %1 = tosa.add %0, %0 : (tensor<91x30xf32>, tensor<91x30xf32>) -> tensor<91x30xf32>
    %2 = tosa.sigmoid %1 : (tensor<91x30xf32>) -> tensor<91x30xf32>
    %3 = tosa.logical_xor %arg2, %arg3 : (tensor<55x9x99x6x81xi1>, tensor<55x1x99x6x81xi1>) -> tensor<55x9x99x6x81xi1>
    return %2, %3 : tensor<91x30xf32>, tensor<55x9x99x6x81xi1>
  }
}
