module {
  func.func @main(%arg0: tensor<88x12x63x13xi32>, %arg1: tensor<82xi1>, %arg2: tensor<57x60x9x13x58x91xf32>, %arg3: tensor<1x60x1x1x58x91xf32>) -> (tensor<88x12x63x13xi32>, tensor<57x60x9x13x58x91xf32>, tensor<1xi1>, tensor<1xi1>, tensor<2xi1>, tensor<57x60x9x13x58x91xf32>, tensor<1xi1>, tensor<1xi1>, tensor<57x60x9x13x58x91xi1>, tensor<57x60x9x13x58x91xi1>, tensor<57x120x9x13x58x91xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<88x12x63x13xi32>) -> tensor<88x12x63x13xi32>
    %1 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<82xi1>) -> tensor<1xi1>
    %2 = tosa.logical_not %1 : (tensor<1xi1>) -> tensor<1xi1>
    %3 = tosa.pow %arg2, %arg3 : (tensor<57x60x9x13x58x91xf32>, tensor<1x60x1x1x58x91xf32>) -> tensor<57x60x9x13x58x91xf32>
    %4 = tosa.maximum %3, %3 : (tensor<57x60x9x13x58x91xf32>, tensor<57x60x9x13x58x91xf32>) -> tensor<57x60x9x13x58x91xf32>
    %t_5 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %5 = tosa.tile %1, %t_5 : (tensor<1xi1>, !tosa.shape<1>) -> tensor<1xi1>
    %6 = tosa.sigmoid %3 : (tensor<57x60x9x13x58x91xf32>) -> tensor<57x60x9x13x58x91xf32>
    %7 = tosa.reciprocal %6 : (tensor<57x60x9x13x58x91xf32>) -> tensor<57x60x9x13x58x91xf32>
    %8 = tosa.clamp %3 {min_val = 5.200000e+01 : f32, max_val = 1.240000e+02 : f32} : (tensor<57x60x9x13x58x91xf32>) -> tensor<57x60x9x13x58x91xf32>
    %9 = tosa.minimum %7, %4 : (tensor<57x60x9x13x58x91xf32>, tensor<57x60x9x13x58x91xf32>) -> tensor<57x60x9x13x58x91xf32>
    %10 = tosa.logical_not %2 : (tensor<1xi1>) -> tensor<1xi1>
    %11 = tosa.reduce_min %5 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.sigmoid %3 : (tensor<57x60x9x13x58x91xf32>) -> tensor<57x60x9x13x58x91xf32>
    %13 = tosa.reciprocal %12 : (tensor<57x60x9x13x58x91xf32>) -> tensor<57x60x9x13x58x91xf32>
    %14 = tosa.logical_not %11 : (tensor<1xi1>) -> tensor<1xi1>
    %t_15 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %15 = tosa.tile %5, %t_15 : (tensor<1xi1>, !tosa.shape<1>) -> tensor<2xi1>
    %16 = tosa.exp %13 : (tensor<57x60x9x13x58x91xf32>) -> tensor<57x60x9x13x58x91xf32>
    %17 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %18 = tosa.bitwise_and %2, %5 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %19 = tosa.greater_equal %13, %3 : (tensor<57x60x9x13x58x91xf32>, tensor<57x60x9x13x58x91xf32>) -> tensor<57x60x9x13x58x91xi1>
    %20 = tosa.bitwise_or %19, %19 : (tensor<57x60x9x13x58x91xi1>, tensor<57x60x9x13x58x91xi1>) -> tensor<57x60x9x13x58x91xi1>
    %21 = tosa.equal %13, %6 : (tensor<57x60x9x13x58x91xf32>, tensor<57x60x9x13x58x91xf32>) -> tensor<57x60x9x13x58x91xi1>
    %22 = tosa.rsqrt %13 : (tensor<57x60x9x13x58x91xf32>) -> tensor<57x60x9x13x58x91xf32>
    %23 = tosa.equal %22, %8 : (tensor<57x60x9x13x58x91xf32>, tensor<57x60x9x13x58x91xf32>) -> tensor<57x60x9x13x58x91xi1>
    %24 = tosa.bitwise_not %19 : (tensor<57x60x9x13x58x91xi1>) -> tensor<57x60x9x13x58x91xi1>
    %25 = tosa.add %23, %23 : (tensor<57x60x9x13x58x91xi1>, tensor<57x60x9x13x58x91xi1>) -> tensor<57x60x9x13x58x91xi1>
    %26 = tosa.concat %24, %21 {axis = 1 : i32} : (tensor<57x60x9x13x58x91xi1>, tensor<57x60x9x13x58x91xi1>) -> tensor<57x120x9x13x58x91xi1>
    return %0, %9, %10, %14, %15, %16, %17, %18, %20, %25, %26 : tensor<88x12x63x13xi32>, tensor<57x60x9x13x58x91xf32>, tensor<1xi1>, tensor<1xi1>, tensor<2xi1>, tensor<57x60x9x13x58x91xf32>, tensor<1xi1>, tensor<1xi1>, tensor<57x60x9x13x58x91xi1>, tensor<57x60x9x13x58x91xi1>, tensor<57x120x9x13x58x91xi1>
  }
}
