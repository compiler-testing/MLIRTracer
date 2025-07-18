module {
  func.func @main(%arg0: tensor<42xi1>, %arg1: tensor<1xi1>, %arg2: tensor<2x79x93xi8>, %arg3: tensor<2x79x93xi8>, %arg4: tensor<2x15x23x54x69xf32>) -> (tensor<84xi1>, tensor<2x15x23x54x69xi1>, tensor<2x15x23x54x69xf32>, tensor<2x15x23x54x69xf32>, tensor<4x237x2xi1>, tensor<84xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<42xi1>, tensor<1xi1>) -> tensor<42xi1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<42xi1>, tensor<42xi1>) -> tensor<42xi1>
    %2 = tosa.concat %1, %1 {axis = 0 : i32} : (tensor<42xi1>, tensor<42xi1>) -> tensor<84xi1>
    %3 = tosa.greater_equal %arg2, %arg3 : (tensor<2x79x93xi8>, tensor<2x79x93xi8>) -> tensor<2x79x93xi1>
    %4 = tosa.logical_not %3 : (tensor<2x79x93xi1>) -> tensor<2x79x93xi1>
    %5 = tosa.identity %2 : (tensor<84xi1>) -> tensor<84xi1>
    %6 = tosa.reduce_any %4 {axis = 2 : i32} : (tensor<2x79x93xi1>) -> tensor<2x79x1xi1>
    %7 = tosa.add %5, %2 : (tensor<84xi1>, tensor<84xi1>) -> tensor<84xi1>
    %8 = tosa.rsqrt %arg4 : (tensor<2x15x23x54x69xf32>) -> tensor<2x15x23x54x69xf32>
    %9 = tosa.ceil %8 : (tensor<2x15x23x54x69xf32>) -> tensor<2x15x23x54x69xf32>
    %10 = tosa.reduce_max %6 {axis = 2 : i32} : (tensor<2x79x1xi1>) -> tensor<2x79x1xi1>
    %11 = tosa.greater_equal %8, %8 : (tensor<2x15x23x54x69xf32>, tensor<2x15x23x54x69xf32>) -> tensor<2x15x23x54x69xi1>
    %12 = tosa.logical_right_shift %11, %11 : (tensor<2x15x23x54x69xi1>, tensor<2x15x23x54x69xi1>) -> tensor<2x15x23x54x69xi1>
    %13 = tosa.exp %8 : (tensor<2x15x23x54x69xf32>) -> tensor<2x15x23x54x69xf32>
    %14 = tosa.sub %8, %9 : (tensor<2x15x23x54x69xf32>, tensor<2x15x23x54x69xf32>) -> tensor<2x15x23x54x69xf32>
    %t_15 = tosa.const_shape {values = dense<[ 2, 3, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %15 = tosa.tile %10, %t_15 : (tensor<2x79x1xi1>, !tosa.shape<3>) -> tensor<4x237x2xi1>
    %16 = tosa.logical_and %5, %2 : (tensor<84xi1>, tensor<84xi1>) -> tensor<84xi1>
    return %7, %12, %13, %14, %15, %16 : tensor<84xi1>, tensor<2x15x23x54x69xi1>, tensor<2x15x23x54x69xf32>, tensor<2x15x23x54x69xf32>, tensor<4x237x2xi1>, tensor<84xi1>
  }
}
