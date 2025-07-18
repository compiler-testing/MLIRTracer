module {
  func.func @main(%arg0: tensor<83x94x38x31xf32>, %arg1: tensor<35x48x86x76x46xi1>, %arg2: tensor<35x1x86x1x1xi1>) -> (tensor<83x94x1x31xf32>, tensor<1x1x2xi32>, tensor<35x48x86x76x46xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 2 : i32} : (tensor<83x94x38x31xf32>) -> tensor<83x94x1x31xf32>
    %1 = tosa.rsqrt %0 : (tensor<83x94x1x31xf32>) -> tensor<83x94x1x31xf32>
    %r_2 = tosa.const_shape {values = dense<[ 120931, 1, 1, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.reshape %1, %r_2 : (tensor<83x94x1x31xf32>, !tosa.shape<4>) -> tensor<120931x1x1x2xf32>
    %3 = tosa.argmax %2 {axis = 0 : i32} : (tensor<120931x1x1x2xf32>) -> tensor<1x1x2xi32>
    %4 = tosa.intdiv %3, %3 : (tensor<1x1x2xi32>, tensor<1x1x2xi32>) -> tensor<1x1x2xi32>
    %5 = tosa.abs %4 : (tensor<1x1x2xi32>) -> tensor<1x1x2xi32>
    %6 = tosa.reciprocal %1 : (tensor<83x94x1x31xf32>) -> tensor<83x94x1x31xf32>
    %7 = tosa.sub %5, %3 : (tensor<1x1x2xi32>, tensor<1x1x2xi32>) -> tensor<1x1x2xi32>
    %8 = tosa.logical_xor %arg1, %arg2 : (tensor<35x48x86x76x46xi1>, tensor<35x1x86x1x1xi1>) -> tensor<35x48x86x76x46xi1>
    return %6, %7, %8 : tensor<83x94x1x31xf32>, tensor<1x1x2xi32>, tensor<35x48x86x76x46xi1>
  }
}
