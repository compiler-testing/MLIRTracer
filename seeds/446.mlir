module {
  func.func @main(%arg0: tensor<48x68x37x92x99x66xf32>, %arg1: tensor<68x51x4x57x60xi1>) -> (tensor<48x68x37x92x99x66xf32>, tensor<68x51x4x57x60xi1>) {
    %0 = tosa.rsqrt %arg0 : (tensor<48x68x37x92x99x66xf32>) -> tensor<48x68x37x92x99x66xf32>
    %1 = tosa.logical_not %arg1 : (tensor<68x51x4x57x60xi1>) -> tensor<68x51x4x57x60xi1>
    return %0, %1 : tensor<48x68x37x92x99x66xf32>, tensor<68x51x4x57x60xi1>
  }
}
