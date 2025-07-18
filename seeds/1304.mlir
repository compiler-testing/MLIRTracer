module {
  func.func @main(%arg0: tensor<22xi16>, %arg1: tensor<73x64x33xf32>, %arg2: tensor<21xi1>, %arg3: tensor<1xi1>, %arg4: tensor<52x12x27x91x17x30xi32>, %arg5: tensor<52x12x1x1x1x30xi32>) -> (tensor<73x64x33xf32>, tensor<1xi16>, tensor<52x12x27x91x17x30xi32>, tensor<73x64x1xf32>, tensor<1xi1>, tensor<73x64x33xf32>, tensor<73x64x33xf32>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<22xi16>) -> tensor<22xi16>
    %1 = tosa.exp %arg1 : (tensor<73x64x33xf32>) -> tensor<73x64x33xf32>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<21xi1>, tensor<1xi1>) -> tensor<21xi1>
    %3 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<22xi16>) -> tensor<1xi16>
    %4 = tosa.clz %3 : (tensor<1xi16>) -> tensor<1xi16>
    %5 = tosa.rsqrt %1 : (tensor<73x64x33xf32>) -> tensor<73x64x33xf32>
    %6 = tosa.clamp %1 {min_val = 5.000000e+00 : f32, max_val = 4.600000e+01 : f32} : (tensor<73x64x33xf32>) -> tensor<73x64x33xf32>
    %7 = tosa.add %6, %6 : (tensor<73x64x33xf32>, tensor<73x64x33xf32>) -> tensor<73x64x33xf32>
    %8 = tosa.log %7 : (tensor<73x64x33xf32>) -> tensor<73x64x33xf32>
    %9 = tosa.clamp %5 {min_val = 5.000000e+00 : f32, max_val = 4.600000e+01 : f32} : (tensor<73x64x33xf32>) -> tensor<73x64x33xf32>
    %10 = tosa.logical_left_shift %3, %4 : (tensor<1xi16>, tensor<1xi16>) -> tensor<1xi16>
    %11 = tosa.intdiv %arg4, %arg5 : (tensor<52x12x27x91x17x30xi32>, tensor<52x12x1x1x1x30xi32>) -> tensor<52x12x27x91x17x30xi32>
    %12 = tosa.reduce_min %8 {axis = 2 : i32} : (tensor<73x64x33xf32>) -> tensor<73x64x1xf32>
    %13 = tosa.reduce_any %2 {axis = 0 : i32} : (tensor<21xi1>) -> tensor<1xi1>
    %14 = tosa.abs %8 : (tensor<73x64x33xf32>) -> tensor<73x64x33xf32>
    %15 = tosa.sigmoid %8 : (tensor<73x64x33xf32>) -> tensor<73x64x33xf32>
    return %9, %10, %11, %12, %13, %14, %15 : tensor<73x64x33xf32>, tensor<1xi16>, tensor<52x12x27x91x17x30xi32>, tensor<73x64x1xf32>, tensor<1xi1>, tensor<73x64x33xf32>, tensor<73x64x33xf32>
  }
}
