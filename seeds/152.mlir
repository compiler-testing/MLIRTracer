module {
  func.func @main(%arg0: tensor<87x91x98x32x96xi1>, %arg1: tensor<1x1x98x1x1xi1>, %arg2: tensor<83x33xf32>, %arg3: tensor<22x65xi1>) -> (tensor<87x91x98x32x96xi1>, tensor<83x33xf32>, tensor<83x33xf32>, tensor<22x1xi1>, tensor<22x1xi1>, tensor<1x1xi1>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<87x91x98x32x96xi1>, tensor<1x1x98x1x1xi1>) -> tensor<87x91x98x32x96xi1>
    %1 = tosa.clz %0 : (tensor<87x91x98x32x96xi1>) -> tensor<87x91x98x32x96xi1>
    %2 = tosa.clz %1 : (tensor<87x91x98x32x96xi1>) -> tensor<87x91x98x32x96xi1>
    %3 = tosa.tanh %arg2 : (tensor<83x33xf32>) -> tensor<83x33xf32>
    %4 = tosa.tanh %3 : (tensor<83x33xf32>) -> tensor<83x33xf32>
    %5 = tosa.reverse %3 {axis = 1 : i32} : (tensor<83x33xf32>) -> tensor<83x33xf32>
    %6 = tosa.reduce_all %arg3 {axis = 1 : i32} : (tensor<22x65xi1>) -> tensor<22x1xi1>
    %7 = tosa.bitwise_or %6, %6 : (tensor<22x1xi1>, tensor<22x1xi1>) -> tensor<22x1xi1>
    %8 = tosa.clz %6 : (tensor<22x1xi1>) -> tensor<22x1xi1>
    %9 = tosa.reduce_sum %6 {axis = 0 : i32} : (tensor<22x1xi1>) -> tensor<1x1xi1>
    %10 = tosa.logical_xor %9, %9 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %11 = tosa.reverse %7 {axis = 0 : i32} : (tensor<22x1xi1>) -> tensor<22x1xi1>
    %12 = tosa.reduce_any %10 {axis = 0 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %13 = tosa.clz %12 : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %14 = tosa.logical_not %13 : (tensor<1x1xi1>) -> tensor<1x1xi1>
    return %2, %4, %5, %8, %11, %14 : tensor<87x91x98x32x96xi1>, tensor<83x33xf32>, tensor<83x33xf32>, tensor<22x1xi1>, tensor<22x1xi1>, tensor<1x1xi1>
  }
}
