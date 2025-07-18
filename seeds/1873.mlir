module {
  func.func @main(%arg0: tensor<56x76x10x23x5x48xf32>, %arg1: tensor<1x76x1x1x1x48xf32>, %arg2: tensor<48xi32>, %arg3: tensor<48xi32>, %arg4: tensor<85xi64>, %arg5: tensor<85xi64>, %arg6: tensor<72x59x68x12x33x30xf32>) -> (tensor<56x76x10x23x5x48xi1>, tensor<48xi1>, tensor<85xi1>, tensor<72x59x68x12x33x30xf32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<56x76x10x23x5x48xf32>, tensor<1x76x1x1x1x48xf32>) -> tensor<56x76x10x23x5x48xi1>
    %1 = tosa.clz %0 : (tensor<56x76x10x23x5x48xi1>) -> tensor<56x76x10x23x5x48xi1>
    %2 = tosa.arithmetic_right_shift %1, %0 {round = true} : (tensor<56x76x10x23x5x48xi1>, tensor<56x76x10x23x5x48xi1>) -> tensor<56x76x10x23x5x48xi1>
    %3 = tosa.equal %arg2, %arg3 : (tensor<48xi32>, tensor<48xi32>) -> tensor<48xi1>
    %4 = tosa.greater %arg4, %arg5 : (tensor<85xi64>, tensor<85xi64>) -> tensor<85xi1>
    %5 = tosa.floor %arg6 : (tensor<72x59x68x12x33x30xf32>) -> tensor<72x59x68x12x33x30xf32>
    %6 = tosa.sigmoid %5 : (tensor<72x59x68x12x33x30xf32>) -> tensor<72x59x68x12x33x30xf32>
    %7 = tosa.ceil %6 : (tensor<72x59x68x12x33x30xf32>) -> tensor<72x59x68x12x33x30xf32>
    return %2, %3, %4, %7 : tensor<56x76x10x23x5x48xi1>, tensor<48xi1>, tensor<85xi1>, tensor<72x59x68x12x33x30xf32>
  }
}
