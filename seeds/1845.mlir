module {
  func.func @main(%arg0: tensor<40x13xi8>, %arg1: tensor<73x68xf32>) -> (tensor<1x1xi8>, tensor<1x136xi1>, tensor<146x204xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<40x13xi8>) -> tensor<40x1xi8>
    %1 = tosa.rsqrt %arg1 : (tensor<73x68xf32>) -> tensor<73x68xf32>
    %2 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<40x1xi8>) -> tensor<1x1xi8>
    %3 = tosa.clamp %2 {min_val = 4 : i8, max_val = 13 : i8} : (tensor<1x1xi8>) -> tensor<1x1xi8>
    %4 = tosa.rsqrt %1 : (tensor<73x68xf32>) -> tensor<73x68xf32>
    %5 = tosa.floor %1 : (tensor<73x68xf32>) -> tensor<73x68xf32>
    %t_6 = tosa.const_shape {values = dense<[ 2, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.tile %5, %t_6 : (tensor<73x68xf32>, !tosa.shape<2>) -> tensor<146x204xf32>
    %7 = tosa.reciprocal %6 : (tensor<146x204xf32>) -> tensor<146x204xf32>
    %8 = tosa.concat %4, %4 {axis = 1 : i32} : (tensor<73x68xf32>, tensor<73x68xf32>) -> tensor<73x136xf32>
    %9 = tosa.reduce_sum %8 {axis = 0 : i32} : (tensor<73x136xf32>) -> tensor<1x136xf32>
    %10 = tosa.clamp %7 {min_val = 1.300000e+01 : f32, max_val = 1.500000e+01 : f32} : (tensor<146x204xf32>) -> tensor<146x204xf32>
    %11 = tosa.equal %9, %9 : (tensor<1x136xf32>, tensor<1x136xf32>) -> tensor<1x136xi1>
    %12 = tosa.equal %10, %10 : (tensor<146x204xf32>, tensor<146x204xf32>) -> tensor<146x204xi1>
    return %3, %11, %12 : tensor<1x1xi8>, tensor<1x136xi1>, tensor<146x204xi1>
  }
}
