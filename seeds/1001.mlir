module {
  func.func @main(%arg0: tensor<12x97x76x21x92x74xf32>, %arg1: tensor<12x97x1x21x92x74xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<85x71xi32>) -> (tensor<12x194x76x21x92x74xf32>, tensor<12x97x76x21x92x74xf32>, tensor<12x97x76x21x92x74xi1>, tensor<i1>, tensor<i1>, tensor<12x97x76x21x92x74xf32>, tensor<1x71xi32>, tensor<1x71xi32>, tensor<1x71xi32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<12x97x76x21x92x74xf32>, tensor<12x97x1x21x92x74xf32>) -> tensor<12x97x76x21x92x74xf32>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tosa.abs %1 : (tensor<i32>) -> tensor<i32>
    %3 = tosa.greater %2, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %4 = tosa.ceil %0 : (tensor<12x97x76x21x92x74xf32>) -> tensor<12x97x76x21x92x74xf32>
    %5 = tosa.concat %4, %4 {axis = 1 : i32} : (tensor<12x97x76x21x92x74xf32>, tensor<12x97x76x21x92x74xf32>) -> tensor<12x194x76x21x92x74xf32>
    %6 = tosa.clz %3 : (tensor<i1>) -> tensor<i1>
    %7 = tosa.floor %0 : (tensor<12x97x76x21x92x74xf32>) -> tensor<12x97x76x21x92x74xf32>
    %8 = tosa.rsqrt %4 : (tensor<12x97x76x21x92x74xf32>) -> tensor<12x97x76x21x92x74xf32>
    %9 = tosa.ceil %8 : (tensor<12x97x76x21x92x74xf32>) -> tensor<12x97x76x21x92x74xf32>
    %10 = tosa.logical_left_shift %3, %6 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %11 = tosa.greater_equal %0, %7 : (tensor<12x97x76x21x92x74xf32>, tensor<12x97x76x21x92x74xf32>) -> tensor<12x97x76x21x92x74xi1>
    %12 = tosa.logical_right_shift %6, %6 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %13 = tosa.logical_not %10 : (tensor<i1>) -> tensor<i1>
    %14 = tosa.bitwise_not %12 : (tensor<i1>) -> tensor<i1>
    %15 = tosa.minimum %4, %0 : (tensor<12x97x76x21x92x74xf32>, tensor<12x97x76x21x92x74xf32>) -> tensor<12x97x76x21x92x74xf32>
    %16 = tosa.reduce_product %arg4 {axis = 0 : i32} : (tensor<85x71xi32>) -> tensor<1x71xi32>
    %17 = tosa.pow %15, %7 : (tensor<12x97x76x21x92x74xf32>, tensor<12x97x76x21x92x74xf32>) -> tensor<12x97x76x21x92x74xf32>
    %18 = tosa.logical_left_shift %16, %16 : (tensor<1x71xi32>, tensor<1x71xi32>) -> tensor<1x71xi32>
    %19 = tosa.bitwise_and %16, %16 : (tensor<1x71xi32>, tensor<1x71xi32>) -> tensor<1x71xi32>
    %20 = tosa.maximum %16, %16 : (tensor<1x71xi32>, tensor<1x71xi32>) -> tensor<1x71xi32>
    return %5, %9, %11, %13, %14, %17, %18, %19, %20 : tensor<12x194x76x21x92x74xf32>, tensor<12x97x76x21x92x74xf32>, tensor<12x97x76x21x92x74xi1>, tensor<i1>, tensor<i1>, tensor<12x97x76x21x92x74xf32>, tensor<1x71xi32>, tensor<1x71xi32>, tensor<1x71xi32>
  }
}
