module {
  func.func @main(%arg0: tensor<74x22x44x66xi1>, %arg1: tensor<23x33x70x76xf32>, %arg2: tensor<91xi32>, %arg3: tensor<1xi32>) -> (tensor<1x22x44x1xi1>, tensor<91xi32>, tensor<23x33x70x76xf32>, tensor<23x33x70x76xf32>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<74x22x44x66xi1>) -> tensor<1x22x44x66xi1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<1x22x44x66xi1>, tensor<1x22x44x66xi1>) -> tensor<1x22x44x66xi1>
    %2 = tosa.reduce_any %1 {axis = 3 : i32} : (tensor<1x22x44x66xi1>) -> tensor<1x22x44x1xi1>
    %3 = tosa.log %arg1 : (tensor<23x33x70x76xf32>) -> tensor<23x33x70x76xf32>
    %4 = tosa.exp %3 : (tensor<23x33x70x76xf32>) -> tensor<23x33x70x76xf32>
    %5 = tosa.bitwise_not %2 : (tensor<1x22x44x1xi1>) -> tensor<1x22x44x1xi1>
    %6 = tosa.intdiv %arg2, %arg3 : (tensor<91xi32>, tensor<1xi32>) -> tensor<91xi32>
    %7 = tosa.clamp %4 {min_val = 2.800000e+01 : f32, max_val = 1.160000e+02 : f32} : (tensor<23x33x70x76xf32>) -> tensor<23x33x70x76xf32>
    %8 = tosa.sigmoid %3 : (tensor<23x33x70x76xf32>) -> tensor<23x33x70x76xf32>
    return %5, %6, %7, %8 : tensor<1x22x44x1xi1>, tensor<91xi32>, tensor<23x33x70x76xf32>, tensor<23x33x70x76xf32>
  }
}
