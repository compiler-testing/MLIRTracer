module {
  func.func @main(%arg0: tensor<84xi1>, %arg1: tensor<76x24x32x87x74xf32>) -> (tensor<1xi1>, tensor<1xi1>, tensor<76x48x32x174x74xi1>, tensor<32x76x24x74x87xf32>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<84xi1>) -> tensor<1xi1>
    %1 = tosa.reciprocal %arg1 : (tensor<76x24x32x87x74xf32>) -> tensor<76x24x32x87x74xf32>
    %2 = tosa.reciprocal %1 : (tensor<76x24x32x87x74xf32>) -> tensor<76x24x32x87x74xf32>
    %3 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.concat %1, %2 {axis = 3 : i32} : (tensor<76x24x32x87x74xf32>, tensor<76x24x32x87x74xf32>) -> tensor<76x24x32x174x74xf32>
    %5 = tosa.clamp %2 {min_val = -2.400000e+01 : f32, max_val = 9.100000e+01 : f32} : (tensor<76x24x32x87x74xf32>) -> tensor<76x24x32x87x74xf32>
    %6 = tosa.tanh %4 : (tensor<76x24x32x174x74xf32>) -> tensor<76x24x32x174x74xf32>
    %7 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %8 = tosa.greater_equal %6, %6 : (tensor<76x24x32x174x74xf32>, tensor<76x24x32x174x74xf32>) -> tensor<76x24x32x174x74xi1>
    %9 = tosa.concat %8, %8 {axis = 1 : i32} : (tensor<76x24x32x174x74xi1>, tensor<76x24x32x174x74xi1>) -> tensor<76x48x32x174x74xi1>
    %10 = "tosa.const"() {values = dense<[2, 0, 1, 4, 3]> : tensor<5xi32>} : () -> tensor<5xi32>
    %11 = tosa.transpose %5 {perms = array<i32: 2, 0, 1, 4, 3>} : (tensor<76x24x32x87x74xf32>) -> tensor<32x76x24x74x87xf32>
    %12 = tosa.abs %9 : (tensor<76x48x32x174x74xi1>) -> tensor<76x48x32x174x74xi1>
    %13 = tosa.pow %11, %11 : (tensor<32x76x24x74x87xf32>, tensor<32x76x24x74x87xf32>) -> tensor<32x76x24x74x87xf32>
    return %3, %7, %12, %13 : tensor<1xi1>, tensor<1xi1>, tensor<76x48x32x174x74xi1>, tensor<32x76x24x74x87xf32>
  }
}
