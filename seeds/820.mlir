module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<51x61x35x39xf32>, %arg3: tensor<51x1x1x1xf32>) -> (tensor<i1>, tensor<51x61x35x39xi1>, tensor<i1>, tensor<51x61x35x39xf32>, tensor<51x61x35x39xi1>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %1 = tosa.logical_xor %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.pow %arg2, %arg3 : (tensor<51x61x35x39xf32>, tensor<51x1x1x1xf32>) -> tensor<51x61x35x39xf32>
    %3 = tosa.bitwise_xor %1, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.sub %3, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.add %4, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %6 = tosa.greater_equal %2, %2 : (tensor<51x61x35x39xf32>, tensor<51x61x35x39xf32>) -> tensor<51x61x35x39xi1>
    %7 = tosa.log %2 : (tensor<51x61x35x39xf32>) -> tensor<51x61x35x39xf32>
    %8 = tosa.clamp %7 {min_val = -4.000000e+01 : f32, max_val = -3.900000e+01 : f32} : (tensor<51x61x35x39xf32>) -> tensor<51x61x35x39xf32>
    %9 = tosa.logical_xor %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %10 = tosa.tanh %8 : (tensor<51x61x35x39xf32>) -> tensor<51x61x35x39xf32>
    %11 = tosa.reciprocal %2 : (tensor<51x61x35x39xf32>) -> tensor<51x61x35x39xf32>
    %12 = tosa.greater_equal %10, %8 : (tensor<51x61x35x39xf32>, tensor<51x61x35x39xf32>) -> tensor<51x61x35x39xi1>
    return %5, %6, %9, %11, %12 : tensor<i1>, tensor<51x61x35x39xi1>, tensor<i1>, tensor<51x61x35x39xf32>, tensor<51x61x35x39xi1>
  }
}
