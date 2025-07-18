module {
  func.func @main(%arg0: tensor<12x97x10x46x88xi32>, %arg1: tensor<12x97x10x1x1xi32>, %arg2: tensor<22x99x96x17xf32>, %arg3: tensor<28x19x20x93x97x36xi1>, %arg4: tensor<28x19x1x1x97x36xi1>) -> (tensor<12x97x10x46x88xi32>, tensor<22x99x96x17xf32>, tensor<28x19x20x93x97x36xi1>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<12x97x10x46x88xi32>, tensor<12x97x10x1x1xi32>) -> tensor<12x97x10x46x88xi32>
    %1 = tosa.floor %arg2 : (tensor<22x99x96x17xf32>) -> tensor<22x99x96x17xf32>
    %2 = tosa.exp %1 : (tensor<22x99x96x17xf32>) -> tensor<22x99x96x17xf32>
    %3 = tosa.add %1, %2 : (tensor<22x99x96x17xf32>, tensor<22x99x96x17xf32>) -> tensor<22x99x96x17xf32>
    %4 = tosa.logical_or %arg3, %arg4 : (tensor<28x19x20x93x97x36xi1>, tensor<28x19x1x1x97x36xi1>) -> tensor<28x19x20x93x97x36xi1>
    return %0, %3, %4 : tensor<12x97x10x46x88xi32>, tensor<22x99x96x17xf32>, tensor<28x19x20x93x97x36xi1>
  }
}
